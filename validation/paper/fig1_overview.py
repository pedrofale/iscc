"""Figure 1 and S9 Fig: one structured tumour, many modalities, one ground truth.

Both figures show the showcase tumour (registry R0/seed1: a 170 x 170 primary field with a 170 x 170
metastatic lattice appended). Fig 1 is drawn at the text width and at most half the text height
(165 x 114 mm), so that its text keeps its size in the paper:
  (a) the workflow, as one row: the tissue setup (glands in stroma, the metastatic site and its
      vessel); the grown tumour coloured by clone, with the metastasis as an inset; the sampling (the
      resected primary bisected into a dissociated and a sectioned half, and four multi-region blocks
      with their clone fractions); and a thumbnail of what each assay returns (a WGS depth track, an
      scDNA copy-number heatmap, an scRNA embedding, a Visium spot map);
  (b) selection depends on the niche, as one row: a schematic of the gland, the stroma and the
      metastasis with the rule of each, and the rules that hold in every niche. The regime's values and
      the share of cancer cells carrying each trait in each compartment go to the render numbers.
S9 Fig follows one clone through every modality, on its own page (165 mm wide):
  (a) its true copy-number profile against the tumour's modal profile;
  (b) a WGS track, drawn over the copies of the cells it pools so that the clone's segments enter at
      its cell fraction (on the multi-region block where the clone is largest; the dissociated half,
      where it is 2% of the cells, is reported in the render numbers);
  (c) the scDNA heatmap with its cells boxed and the calls against the truth;
  (d) the scRNA expression shift along the genome against the dosage law on its true copies;
  (e) the scRNA embedding with its cells highlighted;
  (f) the Visium map of its true fraction per spot;
  (g) its place in the genealogy, with the copy-number events on its path and the clones' cells in the
      metastasis.
The strip of measurement biases that Fig 1 used to carry is S2 Fig (``supp_biases.py``).

Definitions:
  * clones: ``common.Clones`` (clades of the true genealogy holding >= 4% of the cancer cells; the
    same partition and colours as Fig 2a, which shows the same tumour);
  * sampling: ``realistic_regime.resect`` resects the PRIMARY only (``Resection``'s default
    compartment, which enumerates demes 0 .. G*G-1 and never the metastatic lattice appended after
    them); ``bisect(0.5, axis="x")`` cuts it at mid-width; the right half is dissociated (WGS of the
    whole dissociated sample; scDNA and scRNA of the cells that survive dissociation, under the default
    cell-type recovery) and the left half is sectioned at half depth for Visium;
  * the clone of S9 Fig: among the clones holding at least 4% (the clone threshold) of the cancer cells
    of each half, so that every assay sees it, the one whose consensus profile differs from the
    tumour's modal profile at the most segments, then the larger;
  * consensus profile: the per-segment copy number most of the clone's cells carry; the modal profile
    is the commonest whole profile among all cancer cells;
  * the assays' technical parameters: WGS and scDNA take the estimates ``estimate_dna`` fits to the
    real references of Fig 3 (GIAB HG002 for bulk WGS, DLP+ OV2295 for single-cell WGS); scRNA takes
    the package's 10x defaults; Visium takes the regime's fit to a real breast-cancer section
    (``regime.VISIUM``);
  * DNA depth is GC-corrected as a caller corrects it (a quadratic in GC per cell or sample, not the
    assay's own efficiency, which is part of the ground truth);
  * scDNA calls: per cell, the corrected coverage is averaged in bins of 50 loci and divided by its
    mean; the cell's scale (ploidy) is the value in 1.5-6 that brings the scaled bins closest to
    integers (Ginkgo's scan), and each bin is called at the rounded value. Calls are scored against the
    cell's true copy number; doublets (true copy number of two cells) are marked and not scored.

Each part is computed once, in its own process, into small caches under
``<OUT>/derived/fig1/fig1_<part>.{json,npz}``; the render reads only the caches and writes every number
it draws (Fig 1 under "a" and "b", S9 Fig under "s9") to ``fig1_render_numbers.json`` beside them.

Usage (from the iscc repository root, with ISCC_PAPER_OUT and ISCC_PAPER_TIER set as for the registry):
    python validation/paper/fig1_overview.py --part tissue          # counts only, about a minute
    python validation/paper/fig1_overview.py --part dissociated     # WGS, scDNA, scRNA of one half
    python validation/paper/fig1_overview.py --part section         # Visium of the other half
    python validation/paper/fig1_overview.py                        # render Fig 1 and S9 Fig from the caches
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import common as C  # noqa: E402
import registry as REG  # noqa: E402
import regime as RG  # noqa: E402
import style as S  # noqa: E402

CACHE = os.path.join(C.DERIVED, "fig1")
SHOWCASE = ("R0", "seed1")
AXIS, DISSOCIATE = "x", "remainder"      # bisect through the largest clone; the right half (the remainder) is dissociated
MIN_FRAC = 0.04                          # the clone threshold of common.Clones, applied within each half
N_SCDNA, BIN = 400, 50                   # scDNA cells (cancer, after recovery); loci per heatmap bin
SECTION_MAX = 50_000                     # cells kept in the Visium section (a binomial subsample per deme)
PLOIDY_SCAN = np.round(np.arange(1.5, 6.0001, 0.01), 2)
DNA_REF = {name: os.path.join(os.path.dirname(HERE), "data", f + ".npz") for name, f in (
    ("HG002", "dna_ref_hg002_bulk_wgs"), ("DLP+", "dna_ref_dlp_sc_wgs"))}
NORMALS = ("epithelial", "stromal", "immune", "host")
NORMAL_COLOUR = {"stromal": "#CDBFA8", "epithelial": S.COMPARTMENT["epithelium"], "immune": "#5E5E5E",
                 "host": "#9DB7CC"}
N_BLOCKS, BLOCK_HALF = 4, 8              # as in Fig 3a: 17 x 17 demes, about 1 mm at the cm scale
FS = 5.5                                 # the smallest text in the paper's figures, in points
FIG1_SIZE = (165, 84)                    # mm: the text width, and at most half the text height (114 mm)
S9_SIZE = (165, 135)                     # mm: S9 Fig at the text width, so that its text keeps its size
MET_BLUE, VESSEL_RED = "#3E7CB1", "#B2182B"


# --- shared readouts -------------------------------------------------------------------------------
def _parent(par, g):
    p = par.get(g)
    return None if p in (None, "None", g) else p


def _blocks(t, k=N_BLOCKS, half=BLOCK_HALF):
    """The four blocks of Fig 3A (``fig3_biases._blocks``), so that both figures cut the same ones."""
    from fig3_biases import _blocks as blocks
    return blocks(t, k=k, half=half)


def clone_tree(t, cl):
    """The clones' genealogy: clade roots joined at their branch points (unary nodes contracted), each
    node with its birth time and the live cancer cells of its subtree in the primary and the metastasis.

    Adapted from validation/paper/fig2_niche.py:clone_tree, without the
    seeding records."""
    par = cl.parents
    below = defaultdict(set)
    top = None
    for r in cl.roots:
        cur = r
        while True:
            p = _parent(par, cur)
            if p is None:
                break
            below[p].add(cur)
            cur = p
        top = cur
    kept = set(cl.roots) | {g for g, ch in below.items() if len(ch) >= 2} | ({top} if top else set())
    memo = {}

    def up(g):
        chain = []
        while g is not None and g not in kept and g not in memo:
            chain.append(g)
            g = _parent(par, g)
        res = memo.get(g, g)
        for c in chain:
            memo[c] = res
        return res

    nodes = {}
    for g in kept:
        p = _parent(par, g)
        rep = t.genotypes.get(g)
        bt = rep.birth_time if rep is not None and getattr(rep, "birth_time", None) is not None else 0.0
        nodes[g] = dict(id=g, parent=up(p) if p is not None else None, time=float(bt),
                        label=cl.label.get(g), prim=0, met=0)
    npd = t.n_primary_demes
    for i, dm in enumerate(t.demes):
        key = "met" if i >= npd else "prim"
        for g, c in dm.items():
            if not t._is_cancer(g):
                continue
            lab = cl.of(g)
            n = cl.roots[cl.labels.index(lab)] if lab != C.ANCESTRAL else up(g)
            if n is not None:
                nodes[n][key] += c

    def depth(g):
        k = 0
        while nodes[g]["parent"] is not None:
            g, k = nodes[g]["parent"], k + 1
        return k
    sub = {g: [nodes[g]["prim"], nodes[g]["met"]] for g in nodes}
    for g in sorted(nodes, key=lambda g: -depth(g)):              # children before parents
        p = nodes[g]["parent"]
        if p is not None:
            sub[p][0] += sub[g][0]
            sub[p][1] += sub[g][1]
    for g in nodes:
        nodes[g]["sub_prim"], nodes[g]["sub_met"] = sub[g]
    return dict(nodes=list(nodes.values()), top=top)


def lineage_events(t, cl, root):
    """Copy-number events and the genome doubling on the path from the founder to ``root``, as
    [birth time, kind, segment, copies before, copies after] between consecutive ancestors still in
    the registry (an event between two of them is attributed to the later one)."""
    path, g = [], root
    while g is not None:
        path.append(g)
        g = _parent(cl.parents, g)
    events, prev = [], None
    for g in reversed(path):
        rep = t.genotypes.get(g)
        if rep is None:
            continue
        cns, wgd = [int(x) for x in rep.genome_summary["seg_cns"]], bool(rep.is_wgd)
        bt = float(getattr(rep, "birth_time", 0.0) or 0.0)
        if prev is not None:
            pc, pw = prev
            if wgd and not pw:
                events.append([bt, "WGD", -1, 0, 0])
                pc = [2 * x for x in pc]
            for s, (a, b) in enumerate(zip(pc, cns)):
                if a != b:
                    events.append([bt, "gain" if b > a else "loss", s, a, b])
        prev = (cns, wgd)
    return dict(events=events, path_len=len(path), in_registry=sum(1 for g in path if g in t.genotypes))


def cache_path(name):
    return os.path.join(CACHE, name)


def _write_json(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    with open(cache_path(name), "w") as f:
        json.dump(obj, f)
    print("wrote", cache_path(name), flush=True)


def _write_npz(name, **arrays):
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(cache_path(name), **arrays)
    print("wrote", cache_path(name), flush=True)


def _read_json(name):
    p = cache_path(name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _read_npz(name):
    p = cache_path(name)
    return dict(np.load(p, allow_pickle=False)) if os.path.exists(p) else None


def _halves(t):
    """(dissociated demes, sectioned demes) of the resected primary, cut along ``AXIS`` through the
    centre of the largest clone of the primary, so that one clone lies in both halves and can be
    followed through every modality."""
    import realistic_regime as RR
    cl = C.Clones(t)
    G = t.grid_size
    n, col = Counter(), Counter()
    for di, deme in enumerate(t.demes[:G * G]):
        c0 = t.deme_coords[di][1] if AXIS == "x" else t.deme_coords[di][0]
        for g, c in deme.items():
            if c > 0 and t._is_cancer(g):
                lab = cl.of(g)
                if lab != C.ANCESTRAL:
                    n[lab] += c
                    col[lab] += c * c0
    frac = 0.5
    if n:
        big = max(sorted(n), key=lambda lab: n[lab])
        frac = (col[big] / n[big] + 0.5) / G
    part, rest = RR.resect(t).bisect(frac=frac, axis=AXIS)
    return (rest, part) if DISSOCIATE == "remainder" else (part, rest)


def _normal_code(gid):
    """-1 .. -4 for the normal type of a normal genotype: its subtypes (fibroblasts, endothelial and
    perivascular cells; T and NK, myeloid, B and plasma cells) are coded as stroma or immune."""
    from iscc.integrations import biological_types
    return -1 - NORMALS.index(str(biological_types([gid])[0]))


def _code(lab, li):
    """Integer code of a cell's label: the clone index for cancer cells, -1 .. -4 for the normal types."""
    return li[lab] if lab in li else _normal_code(lab)


# --- part: tissue (counts only) ----------------------------------------------------------------------
def compute_tissue():
    import realistic_regime as RR
    t = REG.load(*SHOWCASE)
    cl = C.Clones(t)
    labels = cl.labels + [C.ANCESTRAL]
    li = {lab: i for i, lab in enumerate(labels)}
    G, npd = int(t.grid_size), int(t.n_primary_demes)
    MG = int(getattr(t, "met_grid_size", 0) or 0)
    gl = np.asarray(t.gland_id)
    lumen = set(int(x) for x in np.concatenate([np.asarray(x) for x in t.gland_lumen_demes])) \
        if t.gland_lumen_demes else set()
    tissue = np.zeros(G * G, np.uint8)                       # 0 stroma, 1 duct wall, 2 lumen
    tissue[gl[:G * G] >= 0] = 1
    tissue[sorted(lumen)] = 2
    outline = [int(i) for i in np.nonzero(gl[:G * G] >= 0)[0]
               if any(nb < G * G and gl[nb] == -1 for nb in t._neighbors(int(i)))]
    diss_demes, sec_demes = _halves(t)
    where = np.zeros(G * G, np.int8)                         # 1 dissociated half, 2 sectioned half
    where[diss_demes], where[sec_demes] = 1, 2
    blocks, centres = _blocks(t)
    block_of = np.full(G * G, -1, np.int8)
    for b, ds in enumerate(blocks):
        block_of[ds] = b

    n_dem = len(t.demes)
    dom = np.full(n_dem, -1, np.int8)                        # dominant clone of each deme
    alpha = np.zeros(n_dem, np.uint8)                        # the deme's cancer share, in %
    counts = defaultdict(Counter)                            # sample -> clone -> cancer cells
    carriers = {k: np.zeros(4) for k in ("gland", "stroma", "met")}
    profiles = defaultdict(Counter)                          # clone -> whole profile -> cells
    wgd = Counter()
    lab_of = {}
    for i, dm in enumerate(t.demes):
        tot, can = 0, Counter()
        for g, c in dm.items():
            tot += c
            if c <= 0 or not t._is_cancer(g):
                continue
            lab = lab_of.get(g)
            if lab is None:
                lab = lab_of[g] = cl.of(g)
            can[lab] += c
            rep = t.genotypes[g]
            profiles[lab][tuple(int(x) for x in rep.genome_summary["seg_cns"])] += c
            wgd[lab] += c * bool(rep.is_wgd)
            ep = rep.evolutionary_parameters
            comp = "met" if i >= npd else ("gland" if gl[i] >= 0 else "stroma")
            carriers[comp] += c * np.array([1.0, ep["breach"] > 0, ep["stromal_survival"] > 0,
                                            ep["met_survival"] > 0])
        if not can:
            continue
        top = can.most_common(1)[0][0]
        dom[i], alpha[i] = li[top], int(round(100 * sum(can.values()) / max(tot, 1)))
        if i >= npd:
            counts["met"].update(can)
            continue
        counts["primary"].update(can)
        counts[{1: "dissociated", 2: "section"}[int(where[i])]].update(can)
        if block_of[i] >= 0:
            counts[f"block {block_of[i] + 1}"].update(can)

    allprof = Counter()
    for lab in profiles:
        allprof.update(profiles[lab])
    modal = [int(x) for x in allprof.most_common(1)[0][0]]
    n_seg = int(t.n_segments)
    consensus, n_diff, top_share, cn_share = {}, {}, {}, {}
    for lab, pr in profiles.items():
        n = sum(pr.values())
        cons, share = [], []
        for s in range(n_seg):
            cs = Counter()
            for prof, c in pr.items():
                cs[prof[s]] += c
            v, c = cs.most_common(1)[0]
            cons.append(int(v))
            share.append(round(c / n, 4))
        consensus[lab], cn_share[lab] = cons, share
        n_diff[lab] = int(sum(a != b for a, b in zip(cons, modal)))
        top_share[lab] = round(pr.most_common(1)[0][1] / n, 4)

    n_half = {h: sum(counts[h].values()) for h in ("dissociated", "section")}
    eligible = [lab for lab in cl.labels
                if all(counts[h][lab] >= MIN_FRAC * n_half[h] for h in ("dissociated", "section"))]
    choice = max(eligible, key=lambda lab: (n_diff[lab], cl.size[lab])) if eligible else None
    root = cl.roots[cl.labels.index(choice)] if choice else None
    lineage = lineage_events(t, cl, root) if root else None
    rel = [e for e in t.events if e.get("event") == "origin_release"]
    sel = t.selection
    out = dict(
        source="/".join(SHOWCASE), grid=G, met_grid=MG, n_primary_demes=npd, time=float(t.time),
        cancer=int(t.get_cancer_size()), n_genotypes=len(t.genotypes),
        release=float(rel[0]["time"]) if rel else None,
        n_glands=int(len(set(gl[:G * G].tolist()) - {-1})),
        founding_deme=int(t.genotypes[t.founder_id].birth_deme),
        met_vessel=[int(x) for x in t.met_vessel] if MG else None,
        labels=labels, colours=[cl.colour[lab] for lab in labels], sizes={k: int(v) for k, v in cl.size.items()},
        counts={k: {lab: int(v.get(lab, 0)) for lab in labels} for k, v in counts.items()},
        carriers={k: dict(cells=int(v[0]), breach=float(v[1] / max(v[0], 1)),
                          stromal_survival=float(v[2] / max(v[0], 1)), met_survival=float(v[3] / max(v[0], 1)))
                  for k, v in carriers.items()},
        wgd_share={lab: round(wgd[lab] / max(sum(profiles[lab].values()), 1), 4) for lab in profiles},
        modal=modal, consensus=consensus, consensus_share=cn_share, n_diff=n_diff,
        top_profile_share=top_share, n_segments=n_seg, segment_sizes=[int(x) for x in sel.segment_sizes],
        axis=AXIS, dissociate=DISSOCIATE, min_frac=MIN_FRAC, eligible=eligible, choice=choice, choice_root=root,
        lineage=lineage, tree=clone_tree(t, cl), outline=outline, block_centres=centres, block_half=BLOCK_HALF,
        # the niche rules' values, read off the tumour (Fig 1b)
        params=dict(stromal_hazard=float(t._stromal_hazard), met_hazard=float(t._met_hazard),
                    transit_floor=float(t._met_transit_floor), met_seed_kappa=float(t._met_seed_kappa),
                    resident_ref=float(t._resident_ref), breach_gated=bool(t._breach_gated_invasion),
                    origin_generations=float(getattr(t, "_origin_generations", float("nan"))),
                    K_duct=int(RR.SPATIAL["K_duct"]), K_stroma=int(RR.SPATIAL["K_stroma"]),
                    stroma_fill=float(RR.SPATIAL["stroma_fill_frac"]), K_met=int(getattr(t, "_cap_met", 0)),
                    host_fill=float(getattr(t, "_host_fill_frac", 0.0)),
                    driver_effect=float(sel.driver_effects), breach_effect=float(sel.breach_effects),
                    stromal_survival_effect=float(sel.stromal_survival_effects),
                    met_survival_effect=float(sel.met_survival_effects),
                    breach_cost=float(sel.breach_cost), stromal_survival_cost=float(sel.stromal_survival_cost),
                    met_survival_cost=float(sel.met_survival_cost), n_breach_genes=int(sel.N_breach),
                    n_stromal_survival_genes=int(sum(len(x) for x in sel.stromal_survival)),
                    n_met_survival_genes=int(sum(len(x) for x in sel.met_survival))))
    _write_json("fig1_tissue.json", out)
    _write_npz("fig1_tissue.npz", tissue=tissue, where=where, block_of=block_of,
               dom_primary=dom[:G * G], alpha_primary=alpha[:G * G],
               dom_met=dom[npd:npd + MG * MG], alpha_met=alpha[npd:npd + MG * MG])
    print("clone of S9 Fig:", choice, "eligible", eligible, {k: n_diff[k] for k in cl.labels}, flush=True)


# --- part: dissociated half (WGS, scDNA, scRNA) -------------------------------------------------------
def _dna_estimate(name, modality):
    """The DNA batch parameters ``estimate_dna`` fits to a real reference (as Fig 3 does)."""
    from iscc.data import estimate_dna
    z = np.load(DNA_REF[name], allow_pickle=True)
    kw = dict(modality=modality, breadth=str(z["breadth"]), depth_model=str(z["depth_model"]))
    for k in ("het_mask", "variant_mask"):
        if k in z.files:
            kw[k] = z[k]
    return estimate_dna(z["coverage"], z["alt"], z["cn"], **kw), str(z["source"])


def _seg_index(columns):
    return np.array([int(str(g).split("_")[1]) for g in columns])


def _bins(seg, size=BIN):
    """Bins of about ``size`` consecutive loci within each segment: (bin of each locus, segment of each bin)."""
    b = np.empty(len(seg), dtype=np.int64)
    bin_seg, k = [], 0
    for s in np.unique(seg):
        idx = np.nonzero(seg == s)[0]
        nb = max(1, int(round(len(idx) / size)))
        for j, chunk in enumerate(np.array_split(idx, nb)):
            b[chunk] = k + j
            bin_seg.append(int(s))
        k += nb
    return b, np.asarray(bin_seg)


def _bin_mean(X, b, nb):
    """Means of the columns of ``X`` (cells x loci) within each bin."""
    out = np.zeros((X.shape[0], nb))
    np.add.at(out.T, b, X.T)
    return out / np.bincount(b, minlength=nb)[None, :]


def _dosage(C_tot, s_g, sat):
    """The engine's total-copy-number dosage law (``Cell.get_exp_alleles`` with ``dosage_law="total"``):
    ``max(1 + s_g (min(C, sat) / 2 - 1), 0)``, zero where ``C == 0``."""
    c = np.minimum(C_tot, sat) if sat is not None else C_tot
    return np.where(C_tot > 0, np.maximum(1.0 + s_g[None, :] * (c / 2.0 - 1.0), 0.0), 0.0)


def umap_coords(counts, seed=0):
    """scanpy's standard embedding of a counts matrix (from the old Fig 1)."""
    import scanpy as sc
    import anndata as ad
    a = ad.AnnData(np.asarray(counts, dtype=np.float32))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=1500)
    a = a[:, a.var.highly_variable].copy()
    sc.pp.pca(a, n_comps=30, random_state=seed)
    sc.pp.neighbors(a, random_state=seed)
    sc.tl.umap(a, random_state=seed)
    return a.obsm["X_umap"]


def _gc_correct(cov, gc, deg=2):
    """Per-row GC correction as a copy-number caller does it (HMMcopy's, with a polynomial for the
    loess): each row's coverage, divided by its mean, over a degree-``deg`` fit against the loci's GC
    content; returns rows of mean 1. The assay's own efficiency (the ground truth) is not used."""
    cov = np.atleast_2d(np.asarray(cov, float))
    out = np.empty_like(cov)
    for i, row in enumerate(cov):
        r = row / row.mean()
        ok = r > 0
        fit = np.polyval(np.polyfit(gc[ok], r[ok], deg), gc)
        out[i] = r / np.maximum(fit, 1e-3)
        out[i] /= out[i].mean()
    return out


def _wgs(cd, lab, is_cancer, k, n_segments, est, seed=11):
    """Bulk WGS of one sample at a fitted estimate: the GC-corrected relative depth per locus, and per
    segment the pool's true mean copy number with and without clone ``k`` (its cells given the mean
    profile of the sample's other cancer cells)."""
    from iscc.data import bulkDNA
    wgs = bulkDNA(breadth=est.breadth, depth_model=est.depth_model, seed=seed, **est.dna_kwargs()).run(cd)
    rel = _gc_correct(wgs.observed_data["coverage"].values, np.asarray(wgs.gc, float))[0]
    seg = _seg_index(cd["cell_cnv"].columns)
    first = np.array([np.nonzero(seg == s)[0][0] for s in range(n_segments)])
    cn_seg = cd["cell_cnv"].values[:, first].astype(float)          # cells x segments (whole-segment events)
    in_k, other = lab == k, is_cancer & (lab != k)
    ref = cn_seg[other].mean(0)
    without = cn_seg.copy()
    without[in_k] = ref
    rec = dict(purity=float(wgs.purity), n_pooled=int(wgs.n_cells_pooled), n_cancer=int(is_cancer.sum()),
               n_choice=int(in_k.sum()), seg_true=cn_seg.mean(0).tolist(), seg_without=without.mean(0).tolist(),
               ref_cn=ref.tolist(), choice_cn=cn_seg[in_k].mean(0).tolist() if in_k.any() else None)
    return rec, rel.astype(np.float32), seg


def _labels(t, cl, cd):
    gid = cd["cell_type"]["cell_id"].astype(str)
    is_cancer = np.array([g in t.genotypes and t._is_cancer(g) for g in gid.values])
    lab = np.array([cl.of(g) if c else g for g, c in zip(gid.values, is_cancer)], dtype=object)
    return gid, is_cancer, lab


def compute_dissociated():
    import gc
    from iscc.data import scDNA, scRNA
    from iscc.sample.dissociation import Dissociation
    import realistic_regime as RR
    tis, tz = _read_json("fig1_tissue.json"), _read_npz("fig1_tissue.npz")
    if tis is None:
        sys.exit("run --part tissue first")
    t = REG.load(*SHOWCASE)
    cl = C.Clones(t)
    assert cl.labels == tis["labels"][:-1], "the clone partition differs from the tissue cache"
    labels, k = tis["labels"], tis["choice"]
    li = {lab: i for i, lab in enumerate(labels)}
    n_segments = int(t.n_segments)
    est_w, src_w = _dna_estimate("HG002", "bulk")
    spec = RR.resect(t)
    # WGS of the multi-region block where clone k is the largest share of the cancer cells
    shares = []
    for b in range(len(tis["block_centres"])):
        cnt = tis["counts"].get(f"block {b + 1}", {})
        shares.append(cnt.get(k, 0) / max(sum(cnt.values()), 1))
    blk = int(np.argmax(shares))
    bcd = spec.dissociate(region=np.nonzero(tz["block_of"] == blk)[0].tolist(), max_cells=RG.T["max_cells"],
                          layers=["cell_snv", "cell_cnv"])
    _, b_cancer, b_lab = _labels(t, cl, bcd)
    wgs_block, block_rel, _ = _wgs(bcd, b_lab, b_cancer, k, n_segments, est_w)
    wgs_block.update(block=blk + 1, share_of_cancer_truth=round(shares[blk], 4))
    del bcd
    diss_demes, _ = _halves(t)
    diss = spec.dissociate(region=diss_demes, max_cells=RG.T["max_cells"], layers=["cell_snv", "cell_cnv", "cell_exp"])
    gid, is_cancer, lab = _labels(t, cl, diss)
    cells = np.asarray(gid.index)
    sat = getattr(t.programs, "dosage_saturation", None)
    law = getattr(t.programs, "dosage_law", None)
    s_g = np.asarray(t.programs.dosage_sensitivity, dtype=float)
    del t, cl, spec
    gc.collect()
    in_k = lab == k
    other = is_cancer & ~in_k

    # WGS of the whole dissociated sample, at the HG002 fit
    wgs_rec, wgs_rel, seg = _wgs(diss, lab, is_cancer, k, n_segments, est_w)
    for rec_ in (wgs_rec, wgs_block):
        rec_.update(reference=src_w, fitted=list(est_w.fitted), mu_depth=float(est_w.hypers.mu_depth),
                    kappa=float(est_w.hypers.kappa))
    gc.collect()

    # the cells that survive dissociation (default cell-type recovery), for scDNA and scRNA
    chosen, meta, _ = Dissociation(diss, rng=np.random.default_rng(3)).run()
    pos = {c: i for i, c in enumerate(cells)}
    rec = np.array([pos[c] for c in chosen])
    rng = np.random.default_rng(0)
    rec_cancer = rec[is_cancer[rec]]
    sc_idx = np.sort(rng.choice(rec_cancer, size=min(N_SCDNA, len(rec_cancer)), replace=False))
    est_s, src_s = _dna_estimate("DLP+", "sc")
    sc = scDNA(n_cells=len(sc_idx), breadth=est_s.breadth, depth_model=est_s.depth_model, seed=21,
               **est_s.dna_kwargs()).run(diss, cell_subset=cells[sc_idx])
    cov = _gc_correct(sc.coverage.values, np.asarray(sc.gc, float))
    b, bin_seg = _bins(_seg_index(sc.coverage.columns))
    sc_rel = _bin_mean(cov, b, len(bin_seg))
    sc_rel = sc_rel / sc_rel.mean(1, keepdims=True)
    sc_true = _bin_mean(sc.true_cn.values.astype(float), b, len(bin_seg))
    sc_doublet = np.asarray(sc.is_doublet, bool)
    sc_rec = dict(reference=src_s, fitted=list(est_s.fitted), mu_depth=float(est_s.hypers.mu_depth),
                  kappa=float(est_s.hypers.kappa), n_cells=int(len(sc_idx)), n_choice=int(in_k[sc_idx].sum()),
                  n_doublets=int(sc_doublet.sum()), bin=BIN)
    del sc, cov
    gc.collect()

    rna = scRNA(n_cells=len(rec), seed=100).run(diss, cell_subset=cells[rec])
    X = rna.observed_counts.values.astype(np.float32)
    del rna
    lib = X.sum(1, keepdims=True)
    Xn = X / np.maximum(lib, 1) * 1e4
    rk, ro = in_k[rec], other[rec]
    m_k, m_o = Xn[rk].mean(0), Xn[ro].mean(0)
    cnv_rec = diss["cell_cnv"].values[rec].astype(float)
    e_k = _dosage(cnv_rec[rk], s_g, sat).mean(0)
    e_o = _dosage(cnv_rec[ro], s_g, sat).mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lfc_exp = np.log2(e_k / e_o)
    del diss, cnv_rec, Xn
    gc.collect()
    xy = umap_coords(X, seed=0)

    out = dict(
        source="/".join(SHOWCASE), choice=k, n_cells=int(len(cells)), n_cancer=int(is_cancer.sum()),
        n_choice=int(in_k.sum()),
        composition={str(x): int(v) for x, v in Counter(["cancer" if c else str(g)
                                                          for g, c in zip(gid.values, is_cancer)]).items()},
        clone_counts={str(x): int(v) for x, v in Counter(lab[is_cancer]).items()},
        wgs=wgs_rec, wgs_block=wgs_block, scdna=sc_rec,
        recovery=dict(n_input=int(meta["n_input"]), n_recovered=int(meta["n_recovered"]),
                      probs=meta["recovery_probs"]),
        scrna=dict(n_cells=int(len(rec)), n_choice=int(rk.sum()), n_other_cancer=int(ro.sum()),
                   median_umi=float(np.median(lib)), protocol="10x"),
        dosage=dict(law=law, saturation=sat, mean_sensitivity=float(s_g.mean())))
    _write_json("fig1_dissociated.json", out)
    _write_npz("fig1_dissociated.npz",
               wgs_rel=wgs_rel, wgs_block_rel=block_rel, locus_seg=seg.astype(np.int16),
               sc_rel=sc_rel.astype(np.float32), sc_true=sc_true.astype(np.float32),
               sc_bin_seg=bin_seg.astype(np.int16), sc_code=np.array([_code(x, li) for x in lab[sc_idx]], np.int8),
               sc_doublet=sc_doublet, umap=xy.astype(np.float32),
               rna_code=np.array([_code(x, li) for x in lab[rec]], np.int8),
               mean_k=m_k.astype(np.float32), mean_o=m_o.astype(np.float32),
               lfc_exp=lfc_exp.astype(np.float32), s_g=s_g.astype(np.float32))


# --- part: sectioned half (Visium) ---------------------------------------------------------------------
def compute_section():
    import gc
    from iscc.data import Visium
    import realistic_regime as RR
    tis = _read_json("fig1_tissue.json")
    if tis is None:
        sys.exit("run --part tissue first")
    t = REG.load(*SHOWCASE)
    cl = C.Clones(t)
    assert cl.labels == tis["labels"][:-1], "the clone partition differs from the tissue cache"
    labels, k = tis["labels"], tis["choice"]
    li = {lab: i for i, lab in enumerate(labels)}
    G = int(t.grid_size)
    _, sec_demes = _halves(t)
    sec = RR.resect(t).slice(region=sec_demes, depth_frac=0.5, max_cells=SECTION_MAX, layers="visium")
    gid = sec["cell_type"]["cell_id"].astype(str)
    code = {}
    for c, g in gid.items():
        code[c] = li[cl.of(g)] if (g in t.genotypes and t._is_cancer(g)) else _normal_code(g)
    del t
    gc.collect()
    vis = Visium(seed=300, **RG.VISIUM).run(sec, grid_side=G)
    n_spots = len(vis.spot_members)
    n_cells = np.zeros(n_spots, np.int32)
    n_cancer = np.zeros(n_spots, np.int32)
    n_k = np.zeros(n_spots, np.int32)
    dom = np.full(n_spots, -1, np.int8)
    n_dom = np.zeros(n_spots, np.int32)
    kk = li[k]
    for s, members in enumerate(vis.spot_members):
        cc = np.array([code[m] for m in members], dtype=int)
        n_cells[s] = len(cc)
        can = cc[cc >= 0]
        n_cancer[s] = len(can)
        n_k[s] = int((can == kk).sum())
        if len(can):
            v, c = np.unique(can, return_counts=True)
            dom[s], n_dom[s] = int(v[c.argmax()]), int(c.max())
    umi = vis.spot_counts.values.sum(1)
    out = dict(source="/".join(SHOWCASE), choice=k, n_section_cells=int(len(gid)), section_max=SECTION_MAX,
               depth_frac=0.5, n_spots=int(n_spots), n_tissue_spots=int((n_cells > 0).sum()),
               n_spots_with_choice=int((n_k > 0).sum()), visium=dict(RG.VISIUM, seed=300, grid_side=G),
               median_cells_per_tissue_spot=float(np.median(n_cells[n_cells > 0])),
               median_umi_tissue=float(np.median(umi[n_cells > 0])),
               median_umi_off_tissue=float(np.median(umi[n_cells == 0])) if (n_cells == 0).any() else None)
    _write_json("fig1_section.json", out)
    _write_npz("fig1_section.npz", coords=np.asarray(vis.spot_coords, np.float32), n_cells=n_cells,
               n_cancer=n_cancer, n_choice=n_k, dom=dom, n_dom=n_dom, umi=umi.astype(np.int32))


# --- render: reads the caches only ------------------------------------------------------------------
def _hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _frame(ax, colour=S.GREY, lw=0.4):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(lw)
        sp.set_color(colour)


def _box(fig, x, y, w, h, fc, ec="none", lw=0.4, r=1.0, z=0):
    """A flat rounded box at a mm position (top-left origin), from the old Fig 1."""
    from matplotlib.patches import FancyBboxPatch
    W, H = fig.get_size_inches() / S.MM
    p = FancyBboxPatch((x / W, 1 - (y + h) / H), w / W, h / H, transform=fig.transFigure,
                       boxstyle=f"round,pad=0,rounding_size={r / W}", fc=fc, ec=ec, lw=lw, zorder=z)
    fig.add_artist(p)
    return p


def _arrow(fig, x0, y0, x1, y1, color="#6F6F6F", lw=0.6):
    from matplotlib.patches import FancyArrowPatch
    W, H = fig.get_size_inches() / S.MM
    fig.add_artist(FancyArrowPatch((x0 / W, 1 - y0 / H), (x1 / W, 1 - y1 / H), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=5, color=color, lw=lw, shrinkA=0, shrinkB=0))


def _pending(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=FS, color=S.GREY, transform=ax.transAxes)
    S.despine(ax, False, False)


def _tissue_rgb(T, Z):
    G = T["grid"]
    tis = Z["tissue"].reshape(G, G)
    rgb = np.zeros((G, G, 3))
    for code, key in ((0, "stroma"), (1, "epithelium"), (2, "lumen")):
        rgb[tis == code] = _hexrgb(S.COMPARTMENT[key])
    return rgb


def _clone_rgb(bg, dom, alpha, colours, outline=None, min_alpha=0.25):
    """The tissue under each deme's dominant clone, blended by the deme's cancer share (as
    ``common.tissue_map`` blends it), with the gland walls drawn over the cancer."""
    rgb = bg.copy()
    G = rgb.shape[0]
    cols = np.array([_hexrgb(c) for c in colours])
    d, a = dom.reshape(G, G), alpha.reshape(G, G) / 100.0
    m = d >= 0
    w = (min_alpha + (1 - min_alpha) * a)[m][:, None]
    rgb[m] = (1 - w) * rgb[m] + w * cols[d[m]]
    if outline:
        rr, cc = np.divmod(np.asarray(outline), G)
        rgb[rr, cc] = 0.5 * rgb[rr, cc] + 0.5 * _hexrgb(S.INK)
    return rgb


def _genome_axis(ax, T, labels=True, top=False):
    """Segment boundaries and numbers on a genome-position axis (loci 0 .. n_genes)."""
    edges = np.concatenate([[0], np.cumsum(T["segment_sizes"])])
    for e in edges[1:-1]:
        ax.axvline(e, color="#D0D0D0", lw=0.35, zorder=0)
    ax.set_xlim(0, edges[-1])
    mids = (edges[:-1] + edges[1:]) / 2
    if labels:
        ax.set_xticks(mids)
        ax.set_xticklabels([str(i + 1) for i in range(len(mids))], fontsize=FS)
        ax.tick_params(axis="x", length=0, pad=1.0, top=top, labeltop=top, bottom=not top, labelbottom=not top)
    else:
        ax.set_xticks([])
    return edges


def call_cn(rel):
    """Integer copy-number calls of single cells: each row (bins, mean 1) scaled by the ploidy in
    1.5-6 that brings it closest to integers (Ginkgo's scan), then rounded."""
    P = PLOIDY_SCAN[None, :, None]
    X = rel[:, None, :] * P
    sos = ((X - np.rint(X)) ** 2).sum(2)
    best = PLOIDY_SCAN[sos.argmin(1)]
    return np.clip(np.rint(rel * best[:, None]), 0, None), best


def _block_means(v, seg, sizes=BIN):
    b, bin_seg = _bins(seg, sizes)
    nb = len(bin_seg)
    s = np.bincount(b, weights=v, minlength=nb)
    n = np.bincount(b, minlength=nb)
    centre = np.bincount(b, weights=np.arange(len(v)), minlength=nb) / n
    return s / n, centre, bin_seg, b


# --- Fig 1A: the workflow, as one row ------------------------------------------------------------------
STAGE_X = dict(tissue=4.0, growth=32.0, sampling=75.0, assays=107.0)
MAP_Y, MAP, TISSUE_MAP = 8.0, 25.0, 22.0  # top and side of the growth and sampling maps; side of the tissue map
ARROW_Y = MAP_Y + MAP / 2
BELOW = MAP_Y + MAP + 3.5                 # the row under the maps: the metastatic site, the clones, the blocks


def render_a(fig, T, Z, D, Dz, V, Vz, nums):
    from matplotlib.patches import Rectangle
    G, MG = T["grid"], T["met_grid"]
    labels, colours = T["labels"], T["colours"]
    col = dict(zip(labels, colours))
    S.panel_letter(fig, "a", 1, 1)
    for i, (key, name) in enumerate((("tissue", "Tissue"), ("growth", "Growth"), ("sampling", "Sampling"),
                                     ("assays", "Assays"))):
        S.text_mm(fig, STAGE_X[key] + (2.5 if i == 0 else 0), 3.3, f"{i + 1}  {name}", fontsize=6.5,
                  fontweight="bold", va="center")
    bg = _tissue_rgb(T, Z)

    # 1. the tissue setup: the primary field, and under it the metastatic site with its vessel
    x0 = STAGE_X["tissue"]
    ax = S.ax_mm(fig, x0, ARROW_Y - TISSUE_MAP / 2, TISSUE_MAP, TISSUE_MAP)
    ax.imshow(bg, interpolation="nearest")
    fr, fc = divmod(T["founding_deme"], G)
    ax.scatter([fc], [fr], s=7, facecolors="none", edgecolors=VESSEL_RED, lw=0.6)
    _frame(ax)
    if MG:
        axm = S.ax_mm(fig, x0, BELOW, 9, 9)
        axm.imshow(np.tile(_hexrgb(S.COMPARTMENT["met"]), (MG, MG, 1)), interpolation="nearest")
        vr, vc = T["met_vessel"]
        axm.scatter([vc], [vr], s=5, color=VESSEL_RED, lw=0)
        _frame(axm, NORMAL_COLOUR["host"], 0.5)
        S.text_mm(fig, x0 + 10.5, BELOW + 4.5, "metastatic\nsite", fontsize=FS, color=MET_BLUE, va="center",
                  linespacing=1.1)
    _arrow(fig, x0 + TISSUE_MAP + 0.8, ARROW_Y, STAGE_X["growth"] - 0.8, ARROW_Y)

    # 2. the grown tumour, coloured by clone, with the metastasis as an inset
    x1 = STAGE_X["growth"]
    grown = _clone_rgb(bg, Z["dom_primary"], Z["alpha_primary"], colours, T["outline"])
    ax = S.ax_mm(fig, x1, MAP_Y, MAP, MAP)
    ax.imshow(grown, interpolation="nearest")
    _frame(ax)
    n_met = sum(T["counts"]["met"].values())
    xi = x1 + MAP + 1.5
    if MG:
        dm = Z["dom_met"].reshape(MG, MG)
        occ = np.argwhere(dm >= 0)
        r0, c0 = occ.min(0) - 4
        r1, c1 = occ.max(0) + 5
        side = int(max(r1 - r0, c1 - c0))
        r0, c0 = int(max(r0, 0)), int(max(c0, 0))
        met_bg = np.tile(_hexrgb(S.COMPARTMENT["met"]), (MG, MG, 1))
        met_rgb = _clone_rgb(met_bg, Z["dom_met"], Z["alpha_met"], colours)[r0:r0 + side, c0:c0 + side]
        axi = S.ax_mm(fig, xi, ARROW_Y - 5, 10, 10)
        axi.imshow(met_rgb, interpolation="nearest")
        vr, vc = T["met_vessel"]
        axi.scatter([vc - c0], [vr - r0], s=4, facecolors="none", edgecolors=VESSEL_RED, lw=0.4)
        _frame(axi, MET_BLUE, 0.6)
        S.text_mm(fig, xi, ARROW_Y - 5.6, "metastasis", fontsize=FS, color=MET_BLUE, va="bottom")
        nums["a"]["met_crop"] = side
    S.text_mm(fig, x1, BELOW + 1.0, "clone", fontsize=FS, color=S.GREY, va="center")
    for i, lab in enumerate(labels):
        row, c = divmod(i, 4)
        xx, yy = x1 + 7.5 + c * 6.0, BELOW + 1.0 + row * 3.2
        _box(fig, xx, yy - 0.9, 1.8, 1.8, col[lab], r=0.3)
        S.text_mm(fig, xx + 2.4, yy, lab.replace("clone ", "") if lab != C.ANCESTRAL else "ancestral",
                  fontsize=FS, va="center")
    _arrow(fig, xi + 11.0, ARROW_Y, STAGE_X["sampling"] - 0.8, ARROW_Y)

    # 3. the sampling: the resected primary cut into a sectioned and a dissociated half, and four blocks
    x2 = STAGE_X["sampling"]
    ax = S.ax_mm(fig, x2, MAP_Y, MAP, MAP)
    ax.imshow(0.55 * grown + 0.45, interpolation="nearest")
    # the cut runs through the centre of the largest clone (``_halves``), not down the middle: the
    # sectioned demes are the columns left of it
    cut = int((Z["where"].reshape(G, G) == 2).any(axis=0).sum())
    c_sec, c_dis = S.MODALITY["spatial"], S.MODALITY["DNA"]
    ax.add_patch(Rectangle((-0.5, -0.5), cut, G, fill=False, ec=c_sec, lw=0.9))
    ax.add_patch(Rectangle((cut - 0.5, -0.5), G - cut, G, fill=False, ec=c_dis, lw=0.9, ls=(0, (2, 1))))
    for b, (r, c) in enumerate(T["block_centres"]):
        h = T["block_half"]
        ax.add_patch(Rectangle((c - h - 0.5, r - h - 0.5), 2 * h + 1, 2 * h + 1, fill=False, ec=S.INK, lw=0.6))
        ax.text(c + h + 2, r - h, str(b + 1), fontsize=FS, color=S.INK, va="top", ha="left")
    _frame(ax, "none")
    S.text_mm(fig, x2, MAP_Y - 0.4, "section", fontsize=FS, color=c_sec, ha="left", va="bottom")
    S.text_mm(fig, x2 + MAP * (cut + G) / (2 * G), MAP_Y - 0.4, "dissociated", fontsize=FS, color=c_dis,
              ha="center", va="bottom")
    nums["a"]["section_columns"] = [0, cut - 1]
    # the blocks' clone fractions (truth, from the counts)
    nb = len(T["block_centres"])
    for b in range(nb):
        cnt = T["counts"].get(f"block {b + 1}", {})
        tot = max(sum(cnt.values()), 1)
        axb = S.ax_mm(fig, x2 + 1 + b * 4.6, BELOW, 3.0, 7.0)
        bottom = 0.0
        for lab in labels:
            f = cnt.get(lab, 0) / tot
            axb.bar(0, f, bottom=bottom, color=col[lab], width=1.0, lw=0)
            bottom += f
        axb.set_xlim(-0.5, 0.5)
        axb.set_ylim(0, 1)
        _frame(axb, "none")
        axb.text(0, -0.06, str(b + 1), transform=axb.get_xaxis_transform(), fontsize=FS, ha="center", va="top")
        nums["a"].setdefault("blocks", {})[str(b + 1)] = {lab: round(cnt.get(lab, 0) / tot, 3) for lab in labels}
    S.text_mm(fig, x2 + 1 + nb * 4.6 + 0.6, BELOW, "True clone\nshares", fontsize=FS, color=S.GREY, va="top",
              linespacing=1.1)
    _arrow(fig, x2 + MAP + 0.8, ARROW_Y, STAGE_X["assays"] - 0.8, ARROW_Y)

    # 4. what each assay returns
    x3 = STAGE_X["assays"]
    right = FIG1_SIZE[0] - 1.0
    title = dict(fontsize=FS, va="bottom")
    S.text_mm(fig, x3, MAP_Y - 0.3, "WGS: depth along the genome", color=S.MODALITY["DNA"], **title)
    row2, h2 = MAP_Y + 13.5, 19.0
    S.text_mm(fig, x3, row2 - 0.7, "scDNA: copy number", color=S.MODALITY["DNA"], **title)
    S.text_mm(fig, x3 + 22.0, row2 - 0.7, "scRNA: cells", color=S.MODALITY["RNA"], **title)
    S.text_mm(fig, x3 + 38.0, row2 - 0.7, "Visium: spots", color=S.MODALITY["spatial"], **title)
    axw = S.ax_mm(fig, x3 + 5.0, MAP_Y + 0.6, right - x3 - 5.0, 7.4)
    axd = S.ax_mm(fig, x3 + 1.6, row2, 17.0, h2)
    axr = S.ax_mm(fig, x3 + 22.0, row2, 14.0, h2)
    axv = S.ax_mm(fig, x3 + 38.0, row2, right - x3 - 38.0, h2)      # about square: a window of the section
    if Dz is None:
        for ax_ in (axw, axd, axr):
            _pending(ax_, "pending")
    else:
        v, centre, bin_seg, _ = _block_means(Dz["wgs_rel"], Dz["locus_seg"])
        axw.scatter(centre, v, s=0.6, color=S.MODALITY["DNA"], lw=0, rasterized=True)
        _genome_axis(axw, T, labels=False)
        axw.set_ylim(0.4, 1.45)
        axw.set_yticks([0.5, 1.0])
        S.despine(axw, True, False)
        calls, _ = call_cn(Dz["sc_rel"].astype(float))
        order = np.lexsort((np.arange(len(calls)), Dz["sc_code"]))
        axd.imshow(calls[order], aspect="auto", cmap=S.DIVERGING, norm=S.cn_norm(6), interpolation="nearest")
        _frame(axd, "none")
        axs = S.ax_mm(fig, x3, row2, 1.1, h2)
        axs.imshow(np.array([_hexrgb(colours[c]) for c in Dz["sc_code"][order]])[:, None, :], aspect="auto",
                   interpolation="nearest")
        _frame(axs, "none")
        xy, rc = Dz["umap"], Dz["rna_code"]
        cc = [colours[c] if c >= 0 else NORMAL_COLOUR[NORMALS[-1 - c]] for c in rc]
        axr.scatter(xy[:, 0], xy[:, 1], s=0.5, c=cc, lw=0, rasterized=True)
        _frame(axr, "none")
    if Vz is None:
        _pending(axv, "pending")
    else:
        nums["a"]["visium_window"] = _visium(axv, Vz, T, mode="clone", square=True)
        _frame(axv, "none")
    nums["a"].update(cancer=T["cancer"], generations=round(T["time"], 1), met_cells=n_met, glands=T["n_glands"],
                     grid=G, met_grid=MG, clone_sizes=T["sizes"])


def _visium(ax, Vz, T, mode="clone", k=None, colour=None, square=False):
    """The Visium spots of the section: by dominant clone (alpha by cancer share) or by the true fraction
    of clone ``k`` among the spot's cells. Off-tissue spots are left out. The section is half the tissue,
    twice as tall as it is wide; ``square`` keeps the square window as wide as the section that holds the
    most cancer cells, and returns its first and last deme row."""
    from matplotlib.colors import LinearSegmentedColormap
    xy, n = Vz["coords"], Vz["n_cells"]
    on = n > 0
    window = None
    if square:
        w = xy[on, 1].max() - xy[on, 1].min()
        rows, cancer = xy[on, 0], Vz["n_cancer"][on]
        top = max(np.unique(rows), key=lambda r: cancer[(rows >= r) & (rows <= r + w)].sum())
        on = on & (xy[:, 0] >= top) & (xy[:, 0] <= top + w)
        window = [int(top), int(top + w)]
    c0, c1 = xy[on, 1].min() - 2, xy[on, 1].max() + 2
    r0, r1 = xy[on, 0].min() - 2, xy[on, 0].max() + 2
    pos, (fw, fh) = ax.get_position(), ax.figure.get_size_inches() / S.MM
    mm_per_deme = min(pos.width * fw / (c1 - c0), pos.height * fh / (r1 - r0))
    d_pt = 2.0 * mm_per_deme * 72 / 25.4 * 0.95               # a spot pitch of 2 demes, in points
    if mode == "clone":
        cols = np.array([_hexrgb(c) for c in T["colours"]])
        rgb = np.tile(_hexrgb(S.COMPARTMENT["stroma"]), (on.sum(), 1))
        dom, frac = Vz["dom"][on], Vz["n_cancer"][on] / n[on]
        m = dom >= 0
        a = (0.25 + 0.75 * frac[m])[:, None]
        rgb[m] = (1 - a) * rgb[m] + a * cols[dom[m]]
        ax.scatter(xy[on, 1], xy[on, 0], s=d_pt ** 2, c=rgb, marker="h", lw=0, rasterized=True)
    else:
        cmap = LinearSegmentedColormap.from_list("k", ["#E4E4E4", colour])
        f = Vz["n_choice"][on] / n[on]
        o = np.argsort(f)
        sc = ax.scatter(xy[on, 1][o], xy[on, 0][o], s=d_pt ** 2, c=f[o], cmap=cmap, vmin=0, vmax=1, marker="h",
                        lw=0, rasterized=True)
        return sc
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.set_aspect("equal", adjustable="box")
    return window


# --- Fig 1B: selection depends on the niche, as one row ----------------------------------------------
NICHE_X, NICHE_SIDE = (4.0, 58.0, 112.0), 22.0
NICHE_RULES = (
    ("Gland", (("Founding acinus", "closed at first"),
               ("Breach trait needed", "to leave the gland"),
               ("Crossing probability", "equal to the trait level"))),
    ("Stroma", (("Death hazard from resident", "stromal cells"),
                ("Hazard rising with their", "share of the deme"),
                ("Hazard offset by", "stromal survival"))),
    ("Metastasis", (("Entry only through", "the vessel"),
                    ("Transit survival rising", "with metastatic survival"),
                    ("Host hazard offset by", "metastatic survival"))),
)


def _cell(ax, x, y, r, fc, ec=None, lw=0.3, z=2):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec or fc, lw=lw, zorder=z))


def _dead(ax, x, y, r):
    """A dead cancer cell: grey, crossed out."""
    _cell(ax, x, y, r, "#BDBDBD", ec="white", z=3)
    ax.scatter([x], [y], marker="x", s=6, color=S.INK, lw=0.5, zorder=4)


def _niche_ax(fig, x, y, background):
    from matplotlib.patches import Rectangle
    ax = S.ax_mm(fig, x, y, NICHE_SIDE, NICHE_SIDE)
    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-14, -14), 28, 28, fc=background, ec="none", zorder=0))
    return ax


def render_b(fig, y0, T, nums):
    from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
    S.panel_letter(fig, "b", 1, y0)
    S.text_mm(fig, 5, y0 + 1.5, "Selection in each niche", fontsize=6.5, fontweight="bold", va="center")
    col = dict(zip(T["labels"], T["colours"]))
    can = [col["clone 1"], col["clone 3"], col["clone 2"]]
    rng = np.random.default_rng(1)
    top = y0 + 4.5

    # the gland: a closed acinus; leaving it needs the breach trait
    ax = _niche_ax(fig, NICHE_X[0], top, S.COMPARTMENT["stroma"])
    ax.add_patch(Circle((0, 0), 10.6, fc="none", ec=S.INK, lw=0.8, zorder=3))         # basement membrane
    for a in np.linspace(0, 2 * np.pi, 19)[:-1]:
        _cell(ax, 9.1 * np.cos(a), 9.1 * np.sin(a), 1.35, S.COMPARTMENT["epithelium"], ec="white")
    ax.add_patch(Circle((0, 0), 7.6, fc=S.COMPARTMENT["lumen"], ec="none", zorder=1))
    for _ in range(26):
        rr, a = 6.3 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        _cell(ax, rr * np.cos(a), rr * np.sin(a), 1.05, can[rng.integers(0, 3)], ec="white")
    _cell(ax, 12.2, 3.0, 1.15, col["clone 3"], ec=S.INK, lw=0.4, z=4)                     # a breaching cell
    ax.add_patch(FancyArrowPatch((6.0, 1.5), (11.6, 2.9), arrowstyle="-|>", mutation_scale=4, color=S.INK,
                                 lw=0.5, zorder=5))

    # the stroma: residents add a hazard that stromal survival offsets
    ax = _niche_ax(fig, NICHE_X[1], top, S.COMPARTMENT["stroma"])
    for _ in range(34):
        x_, y_ = rng.uniform(-13, 13, 2)
        ax.add_patch(Ellipse((x_, y_), 4.2, 1.1, angle=rng.uniform(0, 180), fc=NORMAL_COLOUR["stromal"],
                             ec="#A8957A", lw=0.3, zorder=1))
    for _ in range(14):
        rr, a = 4.2 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        _cell(ax, -1 + rr * np.cos(a), 1 + rr * np.sin(a), 1.05, can[rng.integers(0, 3)], ec="white", z=3)
    _dead(ax, 7.0, -6.0, 1.05)

    # the metastasis: vessel entry with transit survival, a host hazard offset by metastatic survival
    ax = _niche_ax(fig, NICHE_X[2], top, S.COMPARTMENT["met"])
    for xx in np.arange(-12.5, 13, 3.1):
        for yy in np.arange(-12.5, 13, 3.1):
            if abs(yy - 6.5) > 2.2 or xx > 3:
                ax.add_patch(Rectangle((xx - 1.1, yy - 1.1), 2.2, 2.2, fc=NORMAL_COLOUR["host"], ec="white",
                                       lw=0.3, zorder=1))
    ax.add_patch(FancyBboxPatch((-14, 5.2), 16.5, 2.6, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="#E7B3B3", ec=VESSEL_RED, lw=0.5, zorder=2))
    _cell(ax, -10, 6.5, 0.9, col["clone 1"], ec="white", z=3)
    _cell(ax, -5, 6.5, 0.9, col["clone 1"], ec="white", z=3)
    _dead(ax, -1.0, 6.5, 0.9)
    for _ in range(12):
        rr, a = 3.4 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        _cell(ax, 4.5 + rr * np.cos(a), 1.5 + rr * np.sin(a), 1.0, col["clone 1"], ec="white", z=3)
    ax.text(-13.2, 8.6, "vessel", fontsize=FS, color=VESSEL_RED, va="bottom", zorder=3)

    # the rule of each niche, beside its schematic
    line = FS * 1.2 * 1.17 / 72 * 25.4                        # mm per line at linespacing 1.2
    for x, (name, paras) in zip(NICHE_X, NICHE_RULES):
        tx = x + NICHE_SIDE + 2.5
        S.text_mm(fig, tx, top + 0.2, name, fontsize=6.5, fontweight="bold", va="top")
        y = top + 4.2
        for p in paras:
            S.text_mm(fig, tx, y, "\n".join(p), fontsize=FS, va="top", linespacing=1.2)
            y += len(p) * line + 1.1
    yb = top + NICHE_SIDE + 2.0
    _box(fig, NICHE_X[0], yb, FIG1_SIZE[0] - 1.0 - NICHE_X[0], 5.6, S.PASTEL["grey"], r=0.8)
    S.text_mm(fig, NICHE_X[0] + 2.0, yb + 2.8, "In every niche", fontsize=FS, fontweight="bold", va="center")
    S.text_mm(fig, NICHE_X[0] + 19.0, yb + 2.8, "Division rate rising with driver dosage up to a maximum, "
              "and a cost to division for each trait wherever it is carried", fontsize=FS, va="center")
    nums["b"] = dict(params=T["params"], carriers=T["carriers"])
    return yb + 5.6


# --- S9 Fig: one clone in every modality -------------------------------------------------------------
S9_X0, S9_W0 = 17.0, 92.0                 # the genome-aligned column (A to D)
S9_XR = 121.0                             # the right column (E, F)
# (top, height) in mm of the axes of each panel of S9 Fig; letters and keys sit above them
S9_ROW = dict(a=(9.0, 10.0), b=(25.5, 11.5), c=(45.0, 30.0), d=(82.5, 12.5), g=(109.0, 19.0),
              e=(7.5, 35.0), f=(52.5, 37.5))


def _key(ax, entries, **kw):
    """A one-row legend in place of a written key; ``entries`` are (handle, label) pairs. By default it sits
    just above the axes, at its left edge."""
    opts = dict(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=len(entries), fontsize=FS, frameon=False,
                handlelength=1.4, handleheight=0.7, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0,
                borderpad=0.0)
    opts.update(kw)
    return ax.legend([h for h, _ in entries], [lab for _, lab in entries], **opts)


def render_s9(fig, T, D, Dz, V, Vz, nums):
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    k = T["choice"]
    labels, colours = T["labels"], T["colours"]
    col = dict(zip(labels, colours))
    ck = col[k]
    li = {lab: i for i, lab in enumerate(labels)}
    X0, W0, XR = S9_X0, S9_W0, S9_XR
    Wf, Hf = fig.get_size_inches() / S.MM
    cons, modal = np.asarray(T["consensus"][k]), np.asarray(T["modal"])
    altered = np.nonzero(cons != modal)[0]
    edges = np.concatenate([[0], np.cumsum(T["segment_sizes"])])
    nd = dict(clone=k, cells=T["sizes"][k], altered_segments=(altered + 1).tolist(), consensus=cons.tolist(),
              modal=modal.tolist(), consensus_share=T["consensus_share"][k], born=_node_time(T, k))

    def shade(ax):
        for s in altered:
            ax.axvspan(edges[s], edges[s + 1], color=ck, alpha=0.10, lw=0, zorder=0)

    def ylab(ax, s, colour=S.INK):
        ax.set_ylabel(s, fontsize=FS, color=colour, linespacing=1.0, labelpad=1.5)

    def _row(p, width=W0):
        """(top, width, height) of panel ``p``'s axes, for ``S.ax_mm`` after its left edge."""
        return S9_ROW[p][0], width, S9_ROW[p][1]

    def row_key(ax, entries, y):
        """A key at ``y`` mm, from the left edge of the genome column."""
        return _key(ax, entries, loc="center left", bbox_to_anchor=(X0 / Wf, 1 - y / Hf),
                    bbox_transform=fig.transFigure)

    # (A) the true copy-number profile
    S.panel_letter(fig, "a", 1, 1)
    ax = S.ax_mm(fig, X0, *_row("a"))
    _genome_axis(ax, T, labels=True, top=True)
    shade(ax)
    ax.hlines(modal, edges[:-1], edges[1:], color=S.GREY, lw=0.8, linestyles=(0, (2, 1)))
    ax.hlines(cons, edges[:-1], edges[1:], color=ck, lw=1.6)
    top_cn = int(max(cons.max(), modal.max()))
    ax.set_ylim(0, top_cn + 0.6)
    ax.set_yticks(list(range(0, top_cn + 1, 2)))
    ylab(ax, "True copy\nnumber")
    S.despine(ax, True, False)
    row_key(ax, [(Line2D([], [], color=ck, lw=1.6), f"consensus profile of {k}"),
                 (Line2D([], [], color=S.GREY, lw=0.8, ls=(0, (2, 1))), "modal profile of the tumor")], 3.0)

    if Dz is None:
        for p in "bcd":
            _pending(S.ax_mm(fig, X0, *_row(p)), "pending: --part dissociated")
        _pending(S.ax_mm(fig, XR, *_row("e", S9_SIZE[0] - 2.0 - XR)), "pending")
    else:
        # (B) WGS: the depth is the pooled copy number, the sum of every cell's copies, so the clone's
        # segments enter it in proportion to its share of the cells. Shown on the multi-region block where
        # the clone is largest; the dissociated half is reported in the render numbers.
        sizes = np.asarray(T["segment_sizes"], float)
        lseg = Dz["locus_seg"].astype(int)

        def parts(w):
            """Per segment, the pooled mean copy number split into normal cells, the other cancer cells
            and clone k (each its cells' share times their mean copies)."""
            n, nc, nk = w["n_pooled"], w["n_cancer"], w["n_choice"]
            normal = np.full(len(sizes), 2.0 * (n - nc) / n)
            kk = nk / n * np.asarray(w["choice_cn"])
            oth = (nc - nk) / n * np.asarray(w["ref_cn"])
            return normal, oth, kk

        def observed(w, rel):
            scale = np.average(w["seg_true"], weights=sizes)            # the pool's mean copy number
            return np.array([rel[lseg == s_].mean() for s_ in range(len(sizes))]) * scale, scale

        S.panel_letter(fig, "b", 1, S9_ROW["b"][0] - 4.0)
        wb = D["wgs_block"]
        normal, oth, kk = parts(wb)
        obs, scale = observed(wb, Dz["wgs_block_rel"].astype(float))
        ax = S.ax_mm(fig, X0, *_row("b"))
        _genome_axis(ax, T, labels=False)
        for s_ in range(len(sizes)):
            x_ = [edges[s_], edges[s_ + 1]]
            ax.fill_between(x_, 0, normal[s_], color="#E3E3E3", lw=0)
            ax.fill_between(x_, normal[s_], normal[s_] + oth[s_], color="#C4C4C4", lw=0)
            ax.fill_between(x_, normal[s_] + oth[s_], normal[s_] + oth[s_] + kk[s_], color=ck, lw=0)
        v, centre, _, _ = _block_means(Dz["wgs_block_rel"].astype(float), lseg)
        ax.scatter(centre, v * scale, s=0.9, color=S.INK, lw=0, alpha=0.7, rasterized=True, zorder=4)
        ax.set_ylim(0, max((normal + oth + kk).max(), np.percentile(v * scale, 99)) * 1.08)
        ylab(ax, "WGS copy\nnumber", S.MODALITY["DNA"])
        S.despine(ax, True, False)
        fb = wb["n_choice"] / wb["n_pooled"]
        _key(ax, [(Line2D([], [], ls="none", marker="o", ms=1.5, color=S.INK, mew=0), "depth"),
                  (Patch(fc=ck, lw=0), k), (Patch(fc="#C4C4C4", lw=0), "other cancer cells"),
                  (Patch(fc="#E3E3E3", lw=0), "normal cells")], bbox_to_anchor=(0.0, 1.04))
        wh = D["wgs"]
        hn, ho, hk = parts(wh)
        h_obs, h_scale = observed(wh, Dz["wgs_rel"].astype(float))
        tot_b, tot_h = normal + oth + kk, hn + ho + hk
        nd["wgs"] = dict(
            block=dict(block=wb["block"], fraction_of_pool=round(fb, 4),
                       fraction_of_cancer=round(wb["n_choice"] / wb["n_cancer"], 4), purity=round(wb["purity"], 4),
                       choice_copies_per_segment=[round(float(x), 3) for x in kk],
                       expected_copies=[round(float(x), 3) for x in tot_b],
                       observed_copies=[round(float(x), 3) for x in obs],
                       r_observed_expected=round(float(np.corrcoef(obs, tot_b)[0, 1]), 3)),
            dissociated_half=dict(fraction_of_pool=round(wh["n_choice"] / wh["n_pooled"], 4),
                                  fraction_of_cancer=round(wh["n_choice"] / wh["n_cancer"], 4),
                                  purity=round(wh["purity"], 4),
                                  choice_copies_per_segment=[round(float(x), 3) for x in hk],
                                  expected_copies=[round(float(x), 3) for x in tot_h],
                                  observed_copies=[round(float(x), 3) for x in h_obs],
                                  r_observed_expected=round(float(np.corrcoef(h_obs, tot_h)[0, 1]), 3)),
            reference=wb["reference"], mu_depth=wb["mu_depth"], kappa=wb["kappa"])

        # (C) scDNA: integer calls, its cells boxed, scored against the truth
        yc, hc = S9_ROW["c"]
        S.panel_letter(fig, "c", 1, yc - 6.5)
        calls, ploidy = call_cn(Dz["sc_rel"].astype(float))
        truth = np.rint(Dz["sc_true"])
        dbl = Dz["sc_doublet"].astype(bool)
        code = Dz["sc_code"].astype(int)
        key = np.where(code == li[k], -1, code)                       # the clone first
        order = np.lexsort((ploidy, key))
        ok = calls == truth
        sing = ~dbl
        acc_all = float(ok[sing].mean())
        acc_k = float(ok[sing & (code == li[k])].mean()) if (sing & (code == li[k])).any() else float("nan")
        ax = S.ax_mm(fig, X0, yc, W0, hc)
        nb = calls.shape[1]
        norm = S.cn_norm(6)
        ax.imshow(calls[order], aspect="auto", cmap=S.DIVERGING, norm=norm, interpolation="nearest",
                  extent=(0, edges[-1], len(order), 0))
        for e in edges[1:-1]:
            ax.axvline(e, color="white", lw=0.3)
        nk = int((code == li[k]).sum())
        ax.add_patch(Rectangle((0, 0), edges[-1], nk, fill=False, ec=ck, lw=1.1, clip_on=False))
        ax.set_xlim(0, edges[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        axs = S.ax_mm(fig, X0 - 1.7, yc, 1.2, hc)
        axs.imshow(np.array([_hexrgb(colours[c]) for c in code[order]])[:, None, :], aspect="auto",
                   interpolation="nearest")
        _frame(axs, "none")
        axe = S.ax_mm(fig, X0 + W0 + 0.6, yc, 1.6, hc)
        err = 1 - ok.mean(1)
        axe.imshow(err[order][:, None], aspect="auto", cmap="Greys", vmin=0, vmax=0.5, interpolation="nearest")
        _frame(axe, "none")
        axe.text(0.5, -0.02, "Error", transform=axe.transAxes, fontsize=FS, ha="center", va="top",
                 rotation=90, color=S.GREY)              # below the strip: beside it, it met panel F's letter
        for i in np.nonzero(dbl[order])[0]:
            axe.plot([1.4], [i], marker="<", ms=2.0, color=S.INK, mew=0, clip_on=False)
        ylab(ax, "scDNA\ncells", S.MODALITY["DNA"])
        ax.yaxis.labelpad = 4.5
        row_key(ax, [(Patch(fill=False, ec=ck, lw=1.1), k),
                     (Line2D([], [], ls="none", marker="<", ms=2.0, color=S.INK, mew=0), "doublet")], yc - 4.7)
        cax = S.ax_mm(fig, X0 + W0 - 22.0, yc - 5.3, 22.0, 1.1)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=S.DIVERGING), cax=cax, orientation="horizontal")
        cb.set_ticks([0, 2, 4, 6])
        cb.ax.tick_params(labelsize=FS, length=1.2, pad=0.8)
        cb.outline.set_linewidth(0.3)
        S.text_mm(fig, X0 + W0 - 23.5, yc - 4.75, "Copy number", fontsize=FS, ha="right", va="center")
        n_ok, n_sing = int(ok[sing].sum()), int(ok[sing].size)
        nd["scdna"] = dict(cells=int(len(order)), choice_cells=nk, bins=int(nb), doublets=int(dbl.sum()),
                           accuracy_singlets=round(acc_all, 6), accuracy_choice=round(acc_k, 6),
                           singlet_bins_correct=n_ok, singlet_bins=n_sing,
                           accuracy_doublets=round(float(ok[dbl].mean()), 4) if dbl.any() else None,
                           median_ploidy=float(np.median(ploidy)),
                           reference=D["scdna"]["reference"], mu_depth=D["scdna"]["mu_depth"])

        # (D) scRNA: the expression shift on its altered segments
        S.panel_letter(fig, "d", 1, S9_ROW["d"][0] - 4.5)
        ax = S.ax_mm(fig, X0, *_row("d"))
        _genome_axis(ax, T, labels=True)
        shade(ax)
        seg = np.repeat(np.arange(len(sizes)), np.asarray(T["segment_sizes"]))
        mk, mo = Dz["mean_k"].astype(float), Dz["mean_o"].astype(float)
        b, bin_seg = _bins(seg)
        nbins = len(bin_seg)
        sk, so = np.bincount(b, weights=mk, minlength=nbins), np.bincount(b, weights=mo, minlength=nbins)
        se = np.bincount(b, weights=mo * np.exp2(np.nan_to_num(Dz["lfc_exp"].astype(float))), minlength=nbins)
        cen = np.bincount(b, weights=np.arange(len(seg)), minlength=nbins) / np.bincount(b, minlength=nbins)
        lfc_o, lfc_e = np.log2(sk / so), np.log2(se / so)
        same = ~np.isin(bin_seg, altered)
        lfc_o = lfc_o - np.median(lfc_o[same])
        lfc_e = lfc_e - np.median(lfc_e[same])
        cn_ratio = np.log2(np.asarray(D["wgs"]["choice_cn"]) / np.asarray(D["wgs"]["ref_cn"]))
        ax.hlines(cn_ratio, edges[:-1], edges[1:], color=S.GREY, lw=0.7, linestyles=(0, (2, 1)))
        ax.plot(cen, lfc_e, color=S.INK, lw=0.7, drawstyle="steps-mid")
        ax.scatter(cen, lfc_o, s=1.4, color=S.MODALITY["RNA"], lw=0, zorder=3)
        ax.axhline(0, color="#BDBDBD", lw=0.4, zorder=0)
        ax.set_ylim(min(-1.4, np.nanmin(lfc_o) - 0.1), max(0.8, np.nanmax(lfc_o) + 0.1))
        ax.set_xlabel("Segment", fontsize=FS, labelpad=1)
        ylab(ax, "scRNA log$_2$\nfold change", S.MODALITY["RNA"])
        S.despine(ax, True, False)
        r = float(np.corrcoef(lfc_o, lfc_e)[0, 1])
        _key(ax, [(Line2D([], [], ls="none", marker="o", ms=1.5, color=S.MODALITY["RNA"], mew=0), "observed"),
                  (Line2D([], [], color=S.INK, lw=0.7), "dosage law"),
                  (Line2D([], [], color=S.GREY, lw=0.7, ls=(0, (2, 1))), "copy number ratio")],
             bbox_to_anchor=(0.0, 1.04))
        S.stat(ax, f"r = {r:.2f}", "lower left")
        alt_obs = {str(s + 1): round(float(np.median(lfc_o[bin_seg == s])), 3) for s in altered}
        nd["scrna"] = dict(cells=D["scrna"]["n_cells"], choice_cells=D["scrna"]["n_choice"], r_obs_expected=round(r, 3),
                           median_lfc_altered=alt_obs,
                           cn_log2_ratio={str(s + 1): round(float(cn_ratio[s]), 3) for s in altered})

        # (E) the scRNA embedding with its cells highlighted
        S.panel_letter(fig, "e", XR - 4.0, 1)
        ax = S.ax_mm(fig, XR, *_row("e", S9_SIZE[0] - 2.0 - XR))
        xy, rc = Dz["umap"], Dz["rna_code"].astype(int)
        m = rc == li[k]
        ax.scatter(xy[~m, 0], xy[~m, 1], s=0.25, color="#D5D5D5", lw=0, rasterized=True)
        ax.scatter(xy[m, 0], xy[m, 1], s=0.6, color=ck, lw=0, rasterized=True)
        _frame(ax, "none")
        ax.set_title(f"scRNA: {k}", fontsize=6.0, color=S.MODALITY["RNA"], pad=1.5)
        nd["umap"] = dict(cells=int(len(m)), choice_cells=int(m.sum()))

    # (F) Visium: its true fraction per spot
    yf, hf = S9_ROW["f"]
    S.panel_letter(fig, "f", XR - 4.0, yf - 5.5)
    if Vz is None:
        _pending(S.ax_mm(fig, XR + 6.0, yf, 30.0, hf), "pending: --part section")
    else:
        ax = S.ax_mm(fig, XR + 6.0, yf, 30.0, hf)
        sc = _visium(ax, Vz, T, mode="fraction", colour=ck)
        n = Vz["n_cells"]
        on = n > 0
        xy = Vz["coords"]
        ax.set_xlim(xy[on, 1].min() - 2, xy[on, 1].max() + 2)
        ax.set_ylim(xy[on, 0].max() + 2, xy[on, 0].min() - 2)
        ax.set_aspect("equal", adjustable="box")
        _frame(ax, "none")
        ax.set_title(f"Visium: {k}", fontsize=6.0, color=S.MODALITY["spatial"], pad=1.5)
        cax = S.ax_mm(fig, XR + 12.0, yf + hf + 3.0, 18.0, 1.1)
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 0.5, 1])
        cb.ax.tick_params(labelsize=FS, length=1.2, pad=0.8)
        cb.outline.set_linewidth(0.3)
        cb.set_label("True share per spot", fontsize=FS, labelpad=1)
        frac = Vz["n_choice"][on] / n[on]
        nd["visium"] = dict(tissue_spots=int(on.sum()), spots_with_choice=int((Vz["n_choice"][on] > 0).sum()),
                            spots_choice_majority=int((frac > 0.5).sum()),
                            median_cells_per_spot=float(np.median(n[on])), section_cells=V["n_section_cells"])

    # (G) its place in the genealogy
    S.panel_letter(fig, "g", 1, S9_ROW["g"][0] - 5.0)
    ax = S.ax_mm(fig, X0, *_row("g", 133.0))
    nd["genealogy"] = _genealogy(ax, T, k, ck)
    nums["s9"] = nd


def _node_time(T, k):
    for n in T["tree"]["nodes"]:
        if n["label"] == k:
            return n["time"]
    return float("nan")


def _genealogy(ax, T, k, ck):
    """The clones' contracted genealogy against time, the path to clone ``k`` in its colour, with the
    copy-number events and the genome doubling on that path; dots at the tips: cells in the metastasis."""
    nodes = {n["id"]: n for n in T["tree"]["nodes"]}
    kids = defaultdict(list)
    for n in nodes.values():
        if n["parent"] is not None:
            kids[n["parent"]].append(n["id"])
    ypos, nxt = {}, [0]

    def place(g):
        ch = sorted(kids[g], key=lambda c: nodes[c]["time"])
        if not ch:
            ypos[g] = nxt[0]
            nxt[0] += 1
        else:
            for c in ch:
                place(c)
            ypos[g] = float(np.mean([ypos[c] for c in ch]))
    roots = [g for g, n in nodes.items() if n["parent"] is None]
    for r in roots:
        place(r)
    col = dict(zip(T["labels"], T["colours"]))
    end = T["time"]
    target = next(g for g, n in nodes.items() if n["label"] == k)
    path, g = set(), target
    while g is not None:
        path.add(g)
        g = nodes[g]["parent"]
    for g, n in nodes.items():
        on = g in path
        c, lw = (ck, 1.3) if on else ("#9A9A9A", 0.6)
        p = n["parent"]
        if p is not None:
            ax.plot([nodes[p]["time"], n["time"]], [ypos[g]] * 2, color=c, lw=lw, solid_capstyle="butt", zorder=2 + on)
            ax.plot([nodes[p]["time"]] * 2, [ypos[p], ypos[g]], color=c if on and nodes[p]["id"] in path else "#9A9A9A",
                    lw=0.6, zorder=1)
        if not kids[g]:
            ax.plot([n["time"], end], [ypos[g]] * 2, color=col.get(n["label"], "#9A9A9A"), lw=2.4,
                    solid_capstyle="butt", zorder=2)
            ax.text(end + 4, ypos[g], n["label"] or "", fontsize=FS, va="center", color=S.INK)
            if n["sub_met"]:
                ax.scatter([end + 45], [ypos[g]], s=1.5 + 20 * np.sqrt(n["sub_met"] / 3500), color=MET_BLUE,
                           lw=0, clip_on=False)
                ax.text(end + 51, ypos[g], f"{n['sub_met']:,}", fontsize=FS, va="center", color=MET_BLUE)
    if T.get("release") is not None:
        ax.axvline(T["release"], color=S.GREY, lw=0.5, ls=":", zorder=0)
        ax.text(T["release"] - 3, nxt[0] - 0.5, "acinus\nopens", fontsize=FS, color=S.GREY, ha="right", va="bottom",
                linespacing=1.0)
    # the events on the path to clone k, placed on the branch that carries them
    path_nodes = sorted((nodes[g] for g in path), key=lambda n: n["time"])
    ev = T["lineage"]["events"] if T.get("lineage") else []
    placed = []
    for tt, kind, s, a, b in ev:
        yy = ypos[path_nodes[0]["id"]]
        for n in path_nodes:
            if tt >= n["time"]:
                yy = ypos[n["id"]]
        if kind == "WGD":
            ax.scatter([tt], [yy], marker="*", s=22, color=S.INK, lw=0, zorder=6)
            continue
        up = kind == "gain"
        ax.scatter([tt], [yy], marker="^" if up else "v", s=9, color="#A81B20" if up else "#2F479E", lw=0, zorder=6)
        near = sum(1 for t_ in placed if abs(t_[0] - tt) < 7 and t_[1] == yy)
        placed.append((tt, yy))
        off = 0.45 + 0.55 * near
        ax.text(tt, yy + off * (-1 if up else 1), str(s + 1), fontsize=FS, ha="center",
                va="bottom" if up else "top", color="#A81B20" if up else "#2F479E")
    ax.set_xlim(0, end * 1.22)
    ax.set_ylim(nxt[0] - 0.4, -1.2)
    ax.set_yticks([])
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("Generations", fontsize=FS, labelpad=1)
    S.despine(ax, False, True)
    ax.text(end + 48, -0.9, "in the\nmetastasis", fontsize=FS, color=MET_BLUE, ha="center", va="bottom",
            linespacing=1.0)
    from matplotlib.lines import Line2D
    kinds = {e[1] for e in ev}
    entries = [(Line2D([], [], ls="none", marker=m_, ms=ms_, color=c_, mew=0), lab_)
               for kind_, m_, ms_, c_, lab_ in (("gain", "^", 3.0, "#A81B20", "copy number gain"),
                                                ("loss", "v", 3.0, "#2F479E", "copy number loss"),
                                                ("WGD", "*", 4.4, S.INK, "whole-genome duplication"))
               if kind_ in kinds]
    if entries:
        _key(ax, entries)
    return dict(nodes=len(nodes), leaves=nxt[0], events=[[round(e[0], 1), e[1], e[2] + 1, e[3], e[4]] for e in ev],
                met_cells_by_clone={n["label"]: n["sub_met"] for n in nodes.values() if n["label"]})


def render():
    S.setup()
    T, Z = _read_json("fig1_tissue.json"), _read_npz("fig1_tissue.npz")
    if T is None:
        sys.exit("no tissue cache: run --part tissue first")
    D, Dz = _read_json("fig1_dissociated.json"), _read_npz("fig1_dissociated.npz")
    V, Vz = _read_json("fig1_section.json"), _read_npz("fig1_section.npz")
    nums = dict(source=T["source"], a={}, caches=dict(tissue=cache_path("fig1_tissue.json"),
                                                       dissociated=D is not None, section=V is not None))
    # Fig 1: the workflow (A) over the niches (B), at the text width and at most half the text height
    fig = S.figure(*FIG1_SIZE)
    render_a(fig, T, Z, D, Dz, V, Vz, nums)
    render_b(fig, BELOW + 12.0, T, nums)
    path = S.save(fig, "fig1_overview")
    # S9 Fig: one clone in every modality
    fig = S.figure(*S9_SIZE)
    render_s9(fig, T, D, Dz, V, Vz, nums)
    path9 = S.save(fig, "supp_clone_modalities")
    with open(cache_path("fig1_render_numbers.json"), "w") as f:
        json.dump(nums, f, indent=1, default=float)
    print("wrote", path, path9, "and", cache_path("fig1_render_numbers.json"))


PARTS = {"tissue": compute_tissue, "dissociated": compute_dissociated, "section": compute_section}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=sorted(PARTS))
    a = ap.parse_args()
    if a.part:
        PARTS[a.part]()
    else:
        render()


if __name__ == "__main__":
    main()
