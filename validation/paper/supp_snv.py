"""Supplementary figure (SNVs): how well each modality detects a tumour's somatic SNVs, Smart-seq3 against
10x scRNA, and SNV-based clone assignment with cardelino, on the R11 arm-genome tumours of Fig 4.

``run`` (one tumour) samples the tumour exactly as Fig 4c and 4d do (``fig4_chain.sample``: the dissociated
cells for scRNA, scDNA and bulk DNA, the 50,000-cell section for Visium; the section is materialised with
the SNV frames as well, which leaves its cells unchanged) and assays the same cells with:

* scRNA, three arms through ``emit_scrna_reads(..., site_coverage_model=True)`` with ``obs_fidelity=1`` and
  ``dropout_mid=0`` (as Fig 4c's scRNA): the 10x preset, the Smart-seq3 preset, and Smart-seq3 at the 10x
  preset's library size (``mu_lib``, ``sigma_lib``), which separates coverage from depth;
* Visium of the section through ``emit_visium_reads(..., site_coverage_model=True)``, Fig 4d's spot
  settings (``regime.VISIUM``, the human-layer kappa), ``obs_fidelity=1``;
* bulk WGS (``bulkDNA``, the wgs preset) of the dissociated cells;
* scDNA as in Fig 4c (the DLP+ fit, the same 1,500 cells and seed). Its per-locus counts are reads of the
  whole ~500 kb bin (the DLP+ fit's depth is in bin units), not reads over the variant site; they are
  reported as emitted (``scdna_bin``) and thinned to the reads that cover the site (``scdna_site``: each
  read kept with probability ``DLP_READ_LEN`` / bin length), which the figure draws.

The truth: a somatic site is a growth gene mutated (``cell_snv`` > 0) in at least one sampled cancer cell
and in no sampled normal cell; its frequency is the share of the sample's cancer cells carrying it. The
dissociated sample and the section each have their own truth (the passenger overlay is drawn per sample).
A site is detected when the assay's reads pooled over all its cells (spots) hold at least 2 alt reads; the
same count at sites no sampled cell carries is the false-positive share (error floor 0.001). Cell level:
among (carrier cancer cell, site) pairs, the share with at least 1 alt read and the mean reads covering the
site (Visium: spots whose cells carry the site and express it).

cardelino gets the TRUE clone-by-variant matrix (Fig 4c's clones; a variant is in a clone when most of the
clone's dissociated cells carry it; variants that differ between clones) and each scRNA arm's alt (A) and
total (D) counts over Fig 4c's 3,000 RNA cells; each cell goes to its most probable clone (ties broken at
random), scored by balanced accuracy against the true clones, chance 1/clones as in fig4_score.py. With
``--fig4-c`` (Fig 4c's per-tumour outputs) it also runs on the clones called from scDNA in Fig 4c, the
matrix taken from the true genotypes of each called clone's DNA cells, each called clone standing for the
true clone most of its DNA cells belong to (Fig 4c's mapping).

Everything a tool reads or writes lives under ``--work`` (node scratch); per tumour only ``summary.json``,
``sites.tsv.gz`` and ``cells.tsv.gz`` reach ``--out/<run>_<task>/``. ``figure`` pools the tumours, draws
``figures/paper1/supp_snv.pdf`` and ``.png`` (through ``_paths``: set ``ISCC_PAPER_DIR``) and writes every
drawn number to ``--out/supp_snv_numbers.json``::

    python validation/paper/supp_snv.py run --run R11 --task seed1 --out OUT --work $SNIC_TMP --fig4-c FIG4/c
    python validation/paper/supp_snv.py cardelino --run R11 --task seed1 --out OUT --work $SNIC_TMP --fig4-c FIG4/c \
        --keys true_smartseq3_10xdepth --cardelino-iter 2000,6000
    ISCC_PAPER_DIR=PAPER ISCC_PAPER_TIER=full python validation/paper/supp_snv.py figure --out OUT [--no-numbers]

``figure`` reads only the stored per-tumour outputs; ISCC_PAPER_TIER must be ``full`` or the figure is stamped as a
preview (style.save).
"""
import argparse
import glob
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

import fig4_chain as FC  # noqa: E402

NORMAL, OTHER = FC.NORMAL, FC.OTHER
SECTION_LAYERS = ("cell_exp", "cell_snv", "cell_rna_vaf")   # what Visium, its allele counts and the truth read
ERROR_RATE = 0.001                                         # the reads' error floor (emit_*_reads default)
MIN_ALT_SITE = 2                                           # pooled alt reads that call a site
MIN_ALT_CELL = 1                                           # alt reads that see a site in one cell
FREQ_EDGES = (0.01, 0.05, 0.2, 0.5)                        # site-frequency bins: <1%, 1-5%, 5-20%, 20-50%, >=50%
FREQ_LABELS = ("<1%", "1-5%", "5-20%", "20-50%", "≥50%")
# DLP+ read length for thinning a bin's reads to those over the site. ASSUMED (150 bp), not taken from the
# DLP+ reference, which carries bin counts only.
DLP_READ_LEN = 150
CARDELINO_ITER = (800, 1200)                               # min_iter, max_iter: the cardelino vignette's
SCRNA_ARMS = ("10x", "smartseq3", "smartseq3_10xdepth")
DNA_ASSAYS = ("bulk", "scdna_bin", "scdna_site")
ASSAYS = SCRNA_ARMS + ("visium",) + DNA_ASSAYS


def _commit():
    return FC._commit()


def _lap(wall, name, t0):
    return FC._lap(wall, name, t0)


# ---------------------------------------------------------------------------------------------------
# sampling and truth
# ---------------------------------------------------------------------------------------------------
def sample(run, task, section_max_cells):
    """``fig4_chain.sample`` with the section's SNV frames built too (same cells: the cell plan does not
    depend on which frames are built)."""
    import realistic_regime as RR
    import regime as RG
    import registry as REG
    from iscc.data.human import HumanGenomeLayer
    t = REG.load(run, task)
    layer = HumanGenomeLayer(t, allow_linear=t.genome_params.get("arms") is None)
    spec = RR.resect(t)
    dis = spec.dissociate(max_cells=RG.T["max_cells"])
    sec = spec.slice(depth_frac=0.5, max_cells=int(section_max_cells), layers=SECTION_LAYERS)
    return t, layer, dis, sec


def site_truth(t, layer, cd):
    """Per gene of one sample: cancer carriers, normal carriers, the frequency among cancer cells, and the
    somatic / non-mutated masks; plus the per-cell carrier matrix and tumour mask."""
    gids = FC._gids(cd)
    tum = np.asarray(layer.is_tumour(gids, t), bool)
    car = cd["cell_snv"].to_numpy() > 0
    n_c = car[tum].sum(axis=0)
    n_n = car[~tum].sum(axis=0)
    freq = n_c / max(int(tum.sum()), 1)
    return dict(gids=gids, tum=tum, car=car, n_cancer_carriers=n_c, n_normal_carriers=n_n, freq=freq,
                somatic=(n_c > 0) & (n_n == 0), nonmut=(n_c == 0) & (n_n == 0), n_cancer=int(tum.sum()),
                n_cells=int(len(tum)))


def freq_bin(freq):
    return np.digitize(freq, FREQ_EDGES)


# ---------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------
def site_level(alt_pooled, truth, extra_masks=None):
    """Counts (not shares) so tumours pool exactly: per frequency bin, true sites and those with
    >= MIN_ALT_SITE pooled alt reads; the same at non-mutated sites."""
    det = np.asarray(alt_pooled) >= MIN_ALT_SITE
    som, non = truth["somatic"], truth["nonmut"]
    b = freq_bin(truth["freq"])
    out = dict(n_sites=int(som.sum()), n_detected=int((det & som).sum()),
               n_nonmut=int(non.sum()), n_false=int((det & non).sum()),
               by_bin=[dict(bin=FREQ_LABELS[k], n_sites=int((som & (b == k)).sum()),
                            n_detected=int((det & som & (b == k)).sum())) for k in range(len(FREQ_LABELS))])
    for name, m in (extra_masks or {}).items():
        out[name] = dict(n_sites=int((som & m).sum()), n_detected=int((det & som & m).sum()),
                         by_bin=[dict(bin=FREQ_LABELS[k], n_sites=int((som & m & (b == k)).sum()),
                                      n_detected=int((det & som & m & (b == k)).sum()))
                                 for k in range(len(FREQ_LABELS))])
    return out


def cell_level(alt, tot, pairs, extra_masks=None):
    """Among the (carrier, site) pairs: how many, how many with >= MIN_ALT_CELL alt reads, and the summed
    reads over the site (mean = sum / n)."""
    def one(m):
        n = int(m.sum())
        return dict(n_pairs=n, n_detected=int(((alt >= MIN_ALT_CELL) & m).sum()),
                    sum_depth=float(tot[m].sum()), n_covered=int(((tot > 0) & m).sum()))
    out = one(pairs)
    for name, g in (extra_masks or {}).items():
        out[name] = one(pairs & g[None, :])
    return out


def _share(n, d):
    return float(n) / d if d else float("nan")


# ---------------------------------------------------------------------------------------------------
# cardelino
# ---------------------------------------------------------------------------------------------------
def clone_config(car, labels, clones):
    """genes x clones: a variant is in a clone when most of the clone's cells carry it (a clone with no
    cell here has no column)."""
    cols, names = [], []
    for c in clones:
        m = labels == c
        if m.any():
            cols.append(car[m].mean(axis=0) > 0.5)
            names.append(c)
    return (np.stack(cols, axis=1) if cols else np.zeros((car.shape[1], 0), bool)), names


def write_cardelino_inputs(d, cfg, cfg_names, var_idx, A, D, cells):
    os.makedirs(d, exist_ok=True)
    vnames = [f"v{g}" for g in var_idx]
    pd.DataFrame(cfg[var_idx].astype(int), index=vnames, columns=cfg_names).to_csv(os.path.join(d, "config.csv"))
    with open(os.path.join(d, "cells.txt"), "w") as f:
        f.write("\n".join(cells) + "\n")
    vj, ck = np.nonzero(D.T > 0)              # D is cells x variants; triplets in variant-major order
    pd.DataFrame({"variant": np.asarray(vnames)[vj], "cell": ck.astype(int),
                  "A": A.T[vj, ck].astype(int), "D": D.T[vj, ck].astype(int)}).to_csv(
        os.path.join(d, "ad.tsv"), sep="\t", index=False)


def launch_cardelino(in_dir, out_dir, seed, iters):
    rs = os.environ.get("ISCC_CARDELINO_RSCRIPT")
    if not rs:
        raise RuntimeError("set ISCC_CARDELINO_RSCRIPT (validation/paper/tools/bin/rscript-cardelino)")
    log = open(out_dir + ".log", "w")
    p = subprocess.Popen([rs, os.path.join(TOOLS, "cardelino_runner.R"), in_dir, out_dir, str(iters[0]),
                          str(iters[1]), str(seed)], stdout=log, stderr=subprocess.STDOUT)
    return p, log


def read_cardelino(out_dir, cfg_names, seed):
    """Most probable clone per cell (exact ties at random), its probability, and the run record."""
    P = pd.read_csv(os.path.join(out_dir, "prob.csv"), index_col=0)
    P.columns = list(cfg_names)
    pr = P.to_numpy(float)
    mx = pr.max(axis=1)
    rng = np.random.default_rng([seed, 0xCD])
    pick = np.array([rng.choice(np.flatnonzero(r >= m - 1e-12)) for r, m in zip(pr, mx)])
    with open(os.path.join(out_dir, "run.json")) as f:
        run = json.load(f)
    return np.asarray(cfg_names, dtype=object)[pick], mx, run


def scrna_arm_kwargs():
    """The three scRNA arms' protocol settings (the rest is shared: obs_fidelity 1, dropout_mid 0, the
    site-coverage model, seed 100 + seed)."""
    from iscc.data.rna import PROTOCOL_PRESETS
    p10 = PROTOCOL_PRESETS["10x"]
    return {"10x": dict(protocol="10x"), "smartseq3": dict(protocol="smartseq3"),
            "smartseq3_10xdepth": dict(protocol="smartseq3", mu_lib=p10["mu_lib"], sigma_lib=p10["sigma_lib"])}


def emit_scrna_arm(dis, dcells, genes, arm, seed, work):
    """One scRNA arm's alt and total counts (cells x genes) through emit_scrna_reads."""
    from iscc.data.reads import emit_scrna_reads
    r = emit_scrna_reads(dis, obs_fidelity=1.0, error_rate=ERROR_RATE, seed=100 + seed, cell_subset=dcells,
                         outdir=os.path.join(work, f"reads_{arm}"), site_coverage_model=True,
                         dropout_mid=0.0, **scrna_arm_kwargs()[arm])
    if list(r["alt"].index) != dcells or list(r["alt"].columns) != genes:
        raise RuntimeError(f"{arm}: cells or genes out of order")
    return r


def cardelino_setup(td, dcells, clone_of, clones, members, a, tag):
    """Fig 4c's RNA cells (fig4_chain.rna_inputs' selection: tumour cells with a true clone, at most --rna-cells of
    them, the same rng), the true clone-by-variant matrix, and (with --fig4-c) the matrix of Fig 4c's scDNA-called
    clones, with the checks that the cells and clones are Fig 4c's."""
    lab_d = FC.label_cells(td["gids"], clone_of)
    keep = np.flatnonzero((lab_d != NORMAL) & (lab_d != OTHER))
    rng = np.random.default_rng([a.seed, 0xCA])
    if keep.size > a.rna_cells:
        keep = np.sort(rng.choice(keep, size=a.rna_cells, replace=False))
    ca_cells = [str(dcells[i]) for i in keep]
    t_fine = lab_d[keep]
    cfg, cfg_names = clone_config(td["car"][td["tum"]], lab_d[td["tum"]], clones)
    var_true = np.flatnonzero(td["somatic"] & cfg.any(axis=1) & ~cfg.all(axis=1)) if cfg.shape[1] else np.array([], int)
    dna_cfg = None
    fig4c = os.path.join(a.fig4_c, tag) if a.fig4_c else None
    checks = {}
    if fig4c and os.path.isdir(fig4c):
        rc = pd.read_csv(os.path.join(fig4c, "rna_cells.tsv.gz"), sep="\t", dtype={"cell": str})
        checks["rna_cells_match_fig4c"] = bool(list(rc["cell"]) == ca_cells and
                                               list(rc["true_clone"].astype(str)) == [str(x) for x in t_fine])
        with open(os.path.join(fig4c, "score.json")) as f:
            checks["true_clones_match_fig4c"] = bool(json.load(f)["true_clones"] == members)
        dc = pd.read_csv(os.path.join(fig4c, "dna_cells.tsv.gz"), sep="\t", dtype={"cell": str})
        dclones = pd.read_csv(os.path.join(fig4c, "dna_clones.tsv"), sep="\t", index_col=0)
        row = {c: i for i, c in enumerate(dcells)}
        called = dc["called_clone"].astype(object).where(dc["called_clone"].notna(), None).to_numpy()
        car_dna = td["car"][[row[c] for c in dc["cell"]]]
        dcfg, dnames = clone_config(car_dna, called, list(dclones.index))
        var_dna = np.flatnonzero(dcfg.any(axis=1) & ~dcfg.all(axis=1)) if dcfg.shape[1] >= 2 else np.array([], int)
        dna_cfg = dict(cfg=dcfg, names=dnames, var=var_dna, mapping=dclones["true_clone"].astype(str).to_dict(),
                       cells=list(dc["cell"]))
    else:
        checks["fig4c"] = f"not given or missing: {fig4c}"
    return dict(keep=keep, ca_cells=ca_cells, t_fine=t_fine, cfg=cfg, cfg_names=cfg_names, var_true=var_true,
                dna_cfg=dna_cfg, checks=checks)


def cardelino_inputs(alt, tot, cs):
    """A and D (cells x variants) of the cardelino cells, for the true and the scDNA-called matrices."""
    keep, var_true, dna_cfg = cs["keep"], cs["var_true"], cs["dna_cfg"]
    D = tot[keep][:, var_true]
    out = dict(A=alt[keep][:, var_true], D=D, no_read=float(np.mean(D.sum(axis=1) == 0)),
               mean_depth=float(D.mean()) if D.size else float("nan"))
    if dna_cfg is not None and dna_cfg["var"].size:
        out["A_dna"] = alt[keep][:, dna_cfg["var"]]
        out["D_dna"] = tot[keep][:, dna_cfg["var"]]
    return out


def score_cardelino(kind, arm, d, cs, inp, seed):
    """One finished cardelino run scored against the true clones: its record and its per-cell columns."""
    t_fine, dna_cfg = cs["t_fine"], cs["dna_cfg"]
    names = cs["cfg_names"] if kind == "true" else dna_cfg["names"]
    key = f"{kind}_{arm}"
    pred, mx, run = read_cardelino(d, names, seed)
    cols = {}
    if kind == "dna":
        cols[f"called_{key}"] = pred
        pred = np.array([dna_cfg["mapping"].get(c) for c in pred], dtype=object)
    sc = FC.score_assignment(t_fine, pred)
    conf = mx > 0.5
    sc_conf = FC.score_assignment(t_fine[conf], pred[conf]) if conf.sum() else {}
    D = inp["D"] if kind == "true" else inp["D_dna"]
    entry = dict(
        balanced_accuracy=sc["balanced_accuracy"], accuracy=sc["accuracy"], ari=sc["ari"], recall=sc["recall"],
        share_prob_above_half=float(conf.mean()),
        confident_accuracy=sc_conf.get("accuracy"), confident_balanced_accuracy=sc_conf.get("balanced_accuracy"),
        share_cells_no_read=inp["no_read"] if kind == "true" else float(np.mean(D.sum(axis=1) == 0)),
        mean_reads_per_cell_variant=inp["mean_depth"] if kind == "true" else float(D.mean()),
        run=run)
    cols[f"pred_{key}"] = pred
    cols[f"prob_{key}"] = mx.round(4)
    return entry, cols


def _print_cardelino(key, r, chance):
    print(f"  cardelino {key}: balanced {r['balanced_accuracy']:.3f} (chance {chance:.3f}), "
          f"p>0.5 {r['share_prob_above_half']:.3f}, no read {r['share_cells_no_read']:.3f}, "
          f"{r['run'].get('n_iter')} iterations, {r['run'].get('wall_s')} s", flush=True)


# ---------------------------------------------------------------------------------------------------
# one tumour
# ---------------------------------------------------------------------------------------------------
def run_tumour(a):
    from iscc.data import bulkDNA, scDNA
    from iscc.data.human import HUMAN_VISIUM_KAPPA
    from iscc.data.reads import emit_visium_reads, site_coverage, site_geometry
    import regime as RG
    tag = f"{a.run}_{a.task}"
    out = os.path.join(a.out, tag)
    work = os.path.join(a.work, f"snv_{tag}")
    os.makedirs(out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    wall, t0 = {}, time.time()
    t, layer, dis, sec = sample(a.run, a.task, a.section_max_cells)
    t0 = _lap(wall, "load_and_sample", t0)
    clone_of, true_prof, members = FC.true_clones(t, layer, [dis, sec])
    clones = list(true_prof.index)
    t0 = _lap(wall, "truth", t0)

    genes = list(dis["cell_snv"].columns)
    if list(sec["cell_snv"].columns) != genes:
        raise RuntimeError("the section's genes differ from the dissociated sample's")
    dcells = list(dis["cell_exp"].index)
    td = site_truth(t, layer, dis)
    ts = site_truth(t, layer, sec)
    length, dist3 = site_geometry(genes)
    in_win = site_coverage(genes, "10x") > 0
    win = {"in_window": in_win, "out_window": ~in_win}
    sites = pd.DataFrame({"gene": [str(g) for g in genes], "freq_dis": td["freq"].round(6),
                          "n_carrier_dis": td["n_cancer_carriers"], "normal_carrier_dis": td["n_normal_carriers"],
                          "somatic_dis": td["somatic"].astype(int), "nonmut_dis": td["nonmut"].astype(int),
                          "freq_sec": ts["freq"].round(6), "somatic_sec": ts["somatic"].astype(int),
                          "nonmut_sec": ts["nonmut"].astype(int), "transcript_len": length.round(0),
                          "site_from_3p": dist3.round(0), "in_3p_window": in_win.astype(int)})
    rec = dict(run=a.run, task=a.task, commit=_commit(), paper_out=os.environ.get("ISCC_PAPER_OUT"),
               tier=os.environ.get("ISCC_PAPER_TIER"), seed=a.seed, section_max_cells=int(a.section_max_cells),
               true_clones=members, n_true_clones=len(clones), chance=1.0 / len(clones),
               dissociated=dict(n_cells=td["n_cells"], n_cancer=td["n_cancer"], n_somatic=int(td["somatic"].sum()),
                                n_nonmut=int(td["nonmut"].sum()),
                                n_normal_carried=int((td["n_normal_carriers"] > 0).sum()),
                                share_somatic_in_window=_share((td["somatic"] & in_win).sum(), td["somatic"].sum())),
               section=dict(n_cells=ts["n_cells"], n_cancer=ts["n_cancer"], n_somatic=int(ts["somatic"].sum()),
                            n_nonmut=int(ts["nonmut"].sum()), n_normal_carried=int((ts["n_normal_carriers"] > 0).sum())),
               n_genes=len(genes), share_genes_in_window=float(in_win.mean()),
               settings=dict(error_rate=ERROR_RATE, min_alt_site=MIN_ALT_SITE, min_alt_cell=MIN_ALT_CELL,
                             freq_edges=list(FREQ_EDGES), dlp_read_len=DLP_READ_LEN,
                             cardelino_iter=list(a.cardelino_iter)),
               assays={})
    pairs_d = td["car"] & td["tum"][:, None] & td["somatic"][None, :]
    t0 = _lap(wall, "site_truth", t0)

    cs = cardelino_setup(td, dcells, clone_of, clones, members, a, tag)
    keep, ca_cells, t_fine, var_true, dna_cfg, checks = (cs["keep"], cs["ca_cells"], cs["t_fine"], cs["var_true"],
                                                         cs["dna_cfg"], cs["checks"])
    cfg, cfg_names = cs["cfg"], cs["cfg_names"]
    sites["cardelino_true_cfg"] = np.isin(np.arange(len(genes)), var_true).astype(int)
    rec["cardelino_config"] = dict(clones=cfg_names, n_variants=int(var_true.size),
                                   n_variants_per_clone={c: int(cfg[var_true, k].sum()) for k, c in enumerate(cfg_names)})
    if dna_cfg is not None:
        rec["cardelino_dna_config"] = dict(clones=dna_cfg["names"], mapping=dna_cfg["mapping"],
                                           n_variants=int(dna_cfg["var"].size),
                                           n_variants_true_cfg_also=int(np.isin(dna_cfg["var"], var_true).sum()))
    t0 = _lap(wall, "configs", t0)

    # ---- scRNA, three arms on the same cells ------------------------------------------------------
    arm_kw = scrna_arm_kwargs()
    rec["scrna_arms"] = {k: dict(v, obs_fidelity=1.0, dropout_mid=0.0, site_coverage_model=True,
                                 seed=100 + a.seed, error_rate=ERROR_RATE) for k, v in arm_kw.items()}
    ca_inputs = {}
    for arm in SCRNA_ARMS:
        r = emit_scrna_arm(dis, dcells, genes, arm, a.seed, work)
        alt, tot = r["alt"].to_numpy(), r["total"].to_numpy()
        sites[f"alt_{arm}"] = alt.sum(axis=0)
        sites[f"tot_{arm}"] = tot.sum(axis=0)
        rec["assays"][arm] = dict(
            site=site_level(sites[f"alt_{arm}"].to_numpy(), td, win),
            cell=cell_level(alt, tot, pairs_d, win),
            mean_site_reads_per_cell=float(tot.sum() / len(dcells)),
            site_cover_values=sorted(set(np.round(np.asarray(r["site_cover"], float), 4).tolist())))
        ca_inputs[arm] = cardelino_inputs(alt, tot, cs)
        del r, alt, tot
        t0 = _lap(wall, f"scrna_{arm}", t0)

    # ---- cardelino, in the background while the other assays run ------------------------------------
    procs = {}
    if len(cfg_names) >= 2 and var_true.size:
        for arm in SCRNA_ARMS:
            d = os.path.join(work, f"ca_true_{arm}")
            write_cardelino_inputs(d + "_in", cfg, cfg_names, var_true, ca_inputs[arm]["A"], ca_inputs[arm]["D"],
                                   ca_cells)
            procs[("true", arm)] = launch_cardelino(d + "_in", d, a.seed + 1, a.cardelino_iter)
    if dna_cfg is not None and len(dna_cfg["names"]) >= 2 and dna_cfg["var"].size:
        for arm in SCRNA_ARMS:
            d = os.path.join(work, f"ca_dna_{arm}")
            write_cardelino_inputs(d + "_in", dna_cfg["cfg"], dna_cfg["names"], dna_cfg["var"],
                                   ca_inputs[arm]["A_dna"], ca_inputs[arm]["D_dna"], ca_cells)
            procs[("dna", arm)] = launch_cardelino(d + "_in", d, a.seed + 1, a.cardelino_iter)
    t0 = _lap(wall, "cardelino_launch", t0)

    # ---- bulk WGS of the dissociated cells -----------------------------------------------------------
    bk = bulkDNA(breadth="wgs", seed=400 + a.seed).run(dis)
    od = bk.observed_data.reindex([g for g in genes])
    sites["alt_bulk"] = od["alt_counts"].fillna(0).astype(int).to_numpy()
    sites["tot_bulk"] = od["coverage"].fillna(0).astype(int).to_numpy()
    rec["assays"]["bulk"] = dict(site=site_level(sites["alt_bulk"].to_numpy(), td, win),
                                 purity=bk.purity, n_cells_pooled=int(bk.n_cells_pooled),
                                 mean_depth=float(sites["tot_bulk"].mean()),
                                 depth_at_somatic=float(sites.loc[td["somatic"], "tot_bulk"].mean()))
    del bk, od
    t0 = _lap(wall, "bulk", t0)

    # ---- scDNA as Fig 4c (DLP+ fit, the same cells and seed) -----------------------------------------
    est = FC.dlp_estimate()
    rng = np.random.default_rng([a.seed, 0xD7A])
    pick = set(rng.choice(np.arange(len(dcells)), size=min(a.n_dna, len(dcells)), replace=False).tolist())
    dna_cells = [c for i, c in enumerate(dcells) if i in pick]
    if dna_cfg is not None:
        checks["dna_cells_match_fig4c"] = bool([str(c) for c in dna_cells] == dna_cfg["cells"])
    dna = scDNA(breadth=est.breadth, depth_model=est.depth_model, seed=200 + a.seed, **est.dna_kwargs()).run(
        dis, cell_subset=dna_cells)
    cov = dna.coverage.reindex(index=dna_cells, columns=genes, fill_value=0).to_numpy().astype(np.int64)
    alt = dna.alt_counts.reindex(index=dna_cells, columns=genes, fill_value=0).to_numpy().astype(np.int64)
    first = layer.grid.sort_values(["growth_gene", "lo"]).groupby("growth_gene").first()
    bin_len = (first["hi"] - first["lo"]).reindex(range(len(genes))).to_numpy(float)
    p_site = np.clip(DLP_READ_LEN / bin_len, 0.0, 1.0)
    rng_t = np.random.default_rng([a.seed, 0x5D7])
    alt_s = rng_t.binomial(alt, p_site[None, :])
    tot_s = alt_s + rng_t.binomial(cov - alt, p_site[None, :])
    row_of = {c: i for i, c in enumerate(dcells)}
    pairs_dna = pairs_d[[row_of[c] for c in dna_cells]]
    for name, (al, to) in (("scdna_bin", (alt, cov)), ("scdna_site", (alt_s, tot_s))):
        sites[f"alt_{name}"] = al.sum(axis=0)
        sites[f"tot_{name}"] = to.sum(axis=0)
        rec["assays"][name] = dict(site=site_level(sites[f"alt_{name}"].to_numpy(), td, win),
                                   cell=cell_level(al, to, pairs_dna, win),
                                   mean_reads_per_locus_per_cell=float(to.mean()), n_cells=len(dna_cells))
    rec["assays"]["scdna_site"]["thinning"] = dict(read_len=DLP_READ_LEN, median_bin_len=float(np.median(bin_len)),
                                                   mean_p_site=float(p_site.mean()))
    rec["assays"]["scdna_bin"]["dna_error_rate"] = float(getattr(dna.hypers, "error_rate", float("nan")))
    del dna, cov, alt, alt_s, tot_s
    t0 = _lap(wall, "scdna", t0)

    # ---- Visium of the section -----------------------------------------------------------------------
    vz = emit_visium_reads(sec, grid_side=t.grid_size, obs_fidelity=1.0, error_rate=ERROR_RATE, seed=300 + a.seed,
                           outdir=os.path.join(work, "visium"), site_coverage_model=True,
                           kappa=HUMAN_VISIUM_KAPPA, **RG.VISIUM)
    if list(vz["alt"].columns) != genes:
        raise RuntimeError("visium: genes out of order")
    alt, tot = vz["alt"].to_numpy(), vz["total"].to_numpy()
    sites["alt_visium"] = alt.sum(axis=0)
    sites["tot_visium"] = tot.sum(axis=0)
    carrier_spot = vz["spot_rna_vaf"].to_numpy() > 0
    pairs_v = carrier_spot & ts["somatic"][None, :]
    n_cells_spot = np.asarray(vz["n_cells"], float)
    rec["assays"]["visium"] = dict(site=site_level(sites["alt_visium"].to_numpy(), ts, win),
                                   cell=cell_level(alt, tot, pairs_v, win), unit="spot",
                                   n_spots=int(len(n_cells_spot)), n_tissue_spots=int((n_cells_spot > 0).sum()),
                                   mean_cells_per_tissue_spot=float(n_cells_spot[n_cells_spot > 0].mean()),
                                   mean_site_reads_per_spot=float(tot.sum() / max((n_cells_spot > 0).sum(), 1)))
    del vz, alt, tot, carrier_spot, pairs_v, sec
    t0 = _lap(wall, "visium", t0)

    # ---- cardelino results ---------------------------------------------------------------------------
    rec["cardelino"] = {}
    cells_tab = pd.DataFrame({"cell": ca_cells, "true_clone": t_fine})
    for (kind, arm), (p, log) in procs.items():
        rc_ = p.wait()
        log.close()
        d = os.path.join(work, f"ca_{kind}_{arm}")
        key = f"{kind}_{arm}"
        if rc_ != 0 or not os.path.exists(os.path.join(d, "prob.csv")):
            rec["cardelino"][key] = dict(failed=True, exit=rc_, log_tail=open(log.name).read()[-3000:])
            print(f"  cardelino {key}: FAILED (exit {rc_})", flush=True)
            continue
        rec["cardelino"][key], cols = score_cardelino(kind, arm, d, cs, ca_inputs[arm], a.seed)
        for c, v in cols.items():
            cells_tab[c] = v
        _print_cardelino(key, rec["cardelino"][key], rec["chance"])
    t0 = _lap(wall, "cardelino_wait", t0)
    rec["checks"] = checks
    rec["wall_s"] = wall

    sites.to_csv(os.path.join(out, "sites.tsv.gz"), sep="\t", index=False)
    cells_tab.to_csv(os.path.join(out, "cells.tsv.gz"), sep="\t", index=False)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rec, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print_summary(rec)


def rerun_cardelino(a):
    """Rerun the named cardelino runs (``--keys``, e.g. true_10x,dna_smartseq3) of one tumour with
    ``--cardelino-iter``, their inputs rebuilt exactly as ``run`` built them (same sample, cells, matrices, scRNA
    seed; the rebuilt inputs are checked against the stored run's), and replace them in the tumour's summary.json
    and cells.tsv.gz. The replaced run's scores and iteration record are kept under ``superseded``."""
    tag = f"{a.run}_{a.task}"
    out = os.path.join(a.out, tag)
    work = os.path.join(a.work, f"snv_{tag}")
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(out, "summary.json")) as f:
        rec = json.load(f)
    keys = [k for k in a.keys.split(",") if k]
    bad = [k for k in keys if k not in rec["cardelino"]]
    if bad:
        sys.exit(f"no such cardelino runs in {tag}: {bad}")
    wall, t0 = {}, time.time()
    t, layer, dis, sec = FC.sample(a.run, a.task, rec["section_max_cells"])   # Fig 4's cut: the same cells
    clone_of, true_prof, members = FC.true_clones(t, layer, [dis, sec])
    del sec
    if members != rec["true_clones"]:
        raise RuntimeError("the true clones differ from the stored run's")
    t0 = _lap(wall, "load_and_sample", t0)
    genes = list(dis["cell_snv"].columns)
    dcells = list(dis["cell_exp"].index)
    td = site_truth(t, layer, dis)
    cs = cardelino_setup(td, dcells, clone_of, list(true_prof.index), members, a, tag)
    cells_tab = pd.read_csv(os.path.join(out, "cells.tsv.gz"), sep="\t", dtype={"cell": str})
    if list(cells_tab["cell"]) != cs["ca_cells"]:
        raise RuntimeError("the cardelino cells differ from the stored run's")
    procs, inputs = {}, {}
    for arm in sorted({k.split("_", 1)[1] for k in keys}):
        r = emit_scrna_arm(dis, dcells, genes, arm, a.seed, work)
        inputs[arm] = cardelino_inputs(r["alt"].to_numpy(), r["total"].to_numpy(), cs)
        del r
    t0 = _lap(wall, "inputs", t0)
    for key in keys:
        kind, arm = key.split("_", 1)
        d = os.path.join(work, f"ca_{key}")
        if kind == "true":
            write_cardelino_inputs(d + "_in", cs["cfg"], cs["cfg_names"], cs["var_true"], inputs[arm]["A"],
                                   inputs[arm]["D"], cs["ca_cells"])
        else:
            dc = cs["dna_cfg"]
            write_cardelino_inputs(d + "_in", dc["cfg"], dc["names"], dc["var"], inputs[arm]["A_dna"],
                                   inputs[arm]["D_dna"], cs["ca_cells"])
        procs[key] = launch_cardelino(d + "_in", d, a.seed + 1, a.cardelino_iter)
    keep_fields = ("balanced_accuracy", "accuracy", "share_prob_above_half", "confident_balanced_accuracy")
    for key, (p, log) in procs.items():
        kind, arm = key.split("_", 1)
        rc_ = p.wait()
        log.close()
        d = os.path.join(work, f"ca_{key}")
        if rc_ != 0 or not os.path.exists(os.path.join(d, "prob.csv")):
            print(f"  cardelino {key}: FAILED (exit {rc_}); the stored run is kept\n" + open(log.name).read()[-3000:],
                  flush=True)
            continue
        entry, cols = score_cardelino(kind, arm, d, cs, inputs[arm], a.seed)
        old = rec["cardelino"][key]
        same = (abs(entry["mean_reads_per_cell_variant"] - old["mean_reads_per_cell_variant"]) < 1e-12
                and abs(entry["share_cells_no_read"] - old["share_cells_no_read"]) < 1e-12)
        if not same:
            raise RuntimeError(f"{key}: the rebuilt inputs differ from the stored run's")
        prev = old.get("superseded")                 # earlier reruns of this run, oldest first
        chain = (prev if isinstance(prev, list) else [prev]) if prev else []
        entry["superseded"] = chain + [dict({f: old.get(f) for f in keep_fields},
                                            n_iter=old["run"].get("n_iter"), min_iter=old["run"].get("min_iter"),
                                            max_iter=old["run"].get("max_iter"),
                                            percent_converged=old["run"].get("percent_converged"),
                                            commit=old.get("rerun_commit", rec.get("commit")))]
        entry["rerun_commit"] = _commit()
        rec["cardelino"][key] = entry
        for c, v in cols.items():
            cells_tab[c] = v
        _print_cardelino(key, entry, rec["chance"])
        print(f"    {entry['run'].get('percent_converged')}% of traces converged (cardelino stops above 99.5%); "
              f"was {old['balanced_accuracy']:.3f} after {old['run'].get('n_iter')} iterations; "
              f"change {entry['balanced_accuracy'] - old['balanced_accuracy']:+.4f}", flush=True)
    t0 = _lap(wall, "cardelino", t0)
    rec.setdefault("cardelino_reruns", []).append(dict(keys=keys, cardelino_iter=list(a.cardelino_iter),
                                                       commit=_commit(), wall_s=wall))
    cells_tab.to_csv(os.path.join(out, "cells.tsv.gz"), sep="\t", index=False)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(rec, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))


def print_summary(rec):
    print("checks:", json.dumps(rec["checks"]), flush=True)
    print(f"sites: dissociated {rec['dissociated']}, section {rec['section']}", flush=True)
    for k, v in rec["assays"].items():
        s = v["site"]
        line = (f"{k:20s} site {_share(s['n_detected'], s['n_sites']):.3f} ({s['n_detected']}/{s['n_sites']}), "
                f"false {_share(s['n_false'], s['n_nonmut']):.4f} ({s['n_false']}/{s['n_nonmut']})")
        if "cell" in v:
            c = v["cell"]
            line += (f"; cell {_share(c['n_detected'], c['n_pairs']):.4f} of {c['n_pairs']} pairs, "
                     f"depth {_share(c['sum_depth'], c['n_pairs']):.3f}")
        print(line, flush=True)


# ---------------------------------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------------------------------
LABEL = {"10x": "10x", "smartseq3": "Smart-seq3", "smartseq3_10xdepth": "Smart-seq3, 10x depth",
         "visium": "Visium", "bulk": "Bulk WGS", "scdna_site": "scDNA",    # scDNA: the reads over the site
         "scdna_bin": "scDNA (bin reads, as emitted)"}
DRAWN = ("10x", "smartseq3", "smartseq3_10xdepth", "visium", "bulk", "scdna_site")
CELL_DRAWN = ("10x", "smartseq3", "smartseq3_10xdepth", "visium", "scdna_site")
TASK_ORDER = ("seed1", "seed2", "disp0.1_seed1", "disp0.9_seed1", "disp0.9_seed2")


def _colours():
    import style as S
    return {"10x": S.MODALITY["RNA"], "smartseq3": "#A63603", "smartseq3_10xdepth": "#FD8D3C",
            "visium": S.MODALITY["spatial"], "bulk": S.MODALITY["DNA"], "scdna_site": "#9E9AC8",
            "scdna_bin": "#CBC9E2"}


def _short(task):
    """A tumor in plain words on two lines, as Fig 4 names it ("dispersal 0.9" over "seed 1")."""
    if task.startswith("disp") and "_seed" in task:
        d, s = task[4:].split("_seed")
        return f"dispersal {d}\nseed {s}"
    if task.startswith("seed"):
        return f"showcase\nseed {task[4:]}"
    return task


def collect(out_dir):
    """Every tumour's summary, ordered as Fig 4 orders them."""
    recs = {}
    for p in glob.glob(os.path.join(out_dir, "*", "summary.json")):
        with open(p) as f:
            r = json.load(f)
        r["_dir"] = os.path.dirname(p)
        recs[r["task"]] = r
    order = [t for t in TASK_ORDER if t in recs] + sorted(t for t in recs if t not in TASK_ORDER)
    return [recs[t] for t in order]


ERR_ALPHA = 1e-3      # the drawn site call: >= MIN_ALT_SITE alt reads AND P(Binomial(total, ERROR_RATE) >= alt) < this


SITE_CALL = (f"error-aware: >= {MIN_ALT_SITE} alt reads pooled over the assay's cells (spots) AND "
             f"P(Binomial(total, {ERROR_RATE}) >= alt) < {ERR_ALPHA} (the error floor)")


def error_aware(recs):
    """The drawn site calls, from each tumour's sites.tsv.gz: >= MIN_ALT_SITE pooled alt reads AND a binomial tail
    below ERR_ALPHA against the error floor (the >= 2-read rule alone is passed on error reads by a deep pooled
    assay). Per assay: ``all``, ``in_window`` and ``out_window`` (the site within 1,000 bases of the 3' end or not),
    each with its frequency bins; and the calls at non-mutated sites."""
    from scipy.stats import binom
    site, spec = {}, {}
    for k in ASSAYS:
        acc = []
        for r in recs:
            s = pd.read_csv(os.path.join(r["_dir"], "sites.tsv.gz"), sep="\t")
            if f"alt_{k}" not in s:
                continue
            alt, tot = s[f"alt_{k}"].to_numpy(), s[f"tot_{k}"].to_numpy()
            call = (alt >= MIN_ALT_SITE) & (binom.sf(alt - 1, tot, ERROR_RATE) < ERR_ALPHA)
            sfx = "sec" if k == "visium" else "dis"
            som, non = s[f"somatic_{sfx}"].to_numpy() == 1, s[f"nonmut_{sfx}"].to_numpy() == 1
            win = s["in_3p_window"].to_numpy() == 1
            acc.append((r["task"], call, som, non, freq_bin(s[f"freq_{sfx}"].to_numpy()), win))
        if not acc:
            continue
        site[k] = {}
        for part in ("all", "in_window", "out_window"):
            def m(w):
                return np.ones_like(w) if part == "all" else (w if part == "in_window" else ~w)
            bins = []
            for kb in range(len(FREQ_LABELS)):
                n = sum(int((som & m(w) & (b == kb)).sum()) for _, call, som, non, b, w in acc)
                d = sum(int((call & som & m(w) & (b == kb)).sum()) for _, call, som, non, b, w in acc)
                bins.append(dict(bin=FREQ_LABELS[kb], n_sites=n, n_detected=d, share=_share(d, n),
                                 per_tumour={t: _share((call & som & m(w) & (b == kb)).sum(), (som & m(w) & (b == kb)).sum())
                                             for t, call, som, non, b, w in acc}))
            n = sum(int((som & m(w)).sum()) for _, call, som, non, b, w in acc)
            d = sum(int((call & som & m(w)).sum()) for _, call, som, non, b, w in acc)
            site[k][part] = dict(n_sites=n, n_detected=d, share=_share(d, n), by_bin=bins,
                                 per_tumour={t: _share((call & som & m(w)).sum(), (som & m(w)).sum())
                                             for t, call, som, non, b, w in acc})
        nn = sum(int(non.sum()) for _, call, som, non, b, w in acc)
        nf = sum(int((call & non).sum()) for _, call, som, non, b, w in acc)
        spec[k] = dict(n_nonmut=nn, n_false=nf, false_share=_share(nf, nn), alpha=ERR_ALPHA, error_rate=ERROR_RATE,
                       per_tumour={t: dict(n_false=int((call & non).sum()), n_nonmut=int(non.sum()))
                                   for t, call, som, non, b, w in acc})
    return site, spec


def numbers(recs):
    """Pooled (counts summed over tumours) and per-tumour values of every drawn quantity."""
    num = dict(tumours=[r["task"] for r in recs], commit=sorted({r["commit"] for r in recs}),
               settings=recs[0]["settings"], n_genes=recs[0]["n_genes"],
               share_genes_in_3p_window=recs[0]["share_genes_in_window"],
               share_somatic_in_3p_window={r["task"]: r["dissociated"]["share_somatic_in_window"] for r in recs},
               sites={r["task"]: dict(dissociated=r["dissociated"], section=r["section"]) for r in recs},
               site_call_drawn=SITE_CALL,
               site_level_min2_only_not_drawn={}, specificity_min2_only_not_drawn={}, cell_level={}, cardelino={})
    for k in ASSAYS:
        rs = [r for r in recs if k in r["assays"]]
        if not rs:
            continue
        s = [r["assays"][k]["site"] for r in rs]
        pooled = {}
        for part in ("all", "in_window", "out_window"):
            src = s if part == "all" else [x[part] for x in s]
            bins = []
            for b in range(len(FREQ_LABELS)):
                n = sum(x["by_bin"][b]["n_sites"] for x in src)
                d = sum(x["by_bin"][b]["n_detected"] for x in src)
                bins.append(dict(bin=FREQ_LABELS[b], n_sites=n, n_detected=d, share=_share(d, n),
                                 per_tumour={r["task"]: _share(x["by_bin"][b]["n_detected"], x["by_bin"][b]["n_sites"])
                                             for r, x in zip(rs, src)}))
            n = sum(x["n_sites"] for x in src)
            d = sum(x["n_detected"] for x in src)
            pooled[part] = dict(n_sites=n, n_detected=d, share=_share(d, n), by_bin=bins,
                                per_tumour={r["task"]: _share(x["n_detected"], x["n_sites"]) for r, x in zip(rs, src)})
        num["site_level_min2_only_not_drawn"][k] = pooled
        nf, nn = sum(x["n_false"] for x in s), sum(x["n_nonmut"] for x in s)
        num["specificity_min2_only_not_drawn"][k] = dict(n_nonmut=nn, n_false=nf, false_share=_share(nf, nn),
                                     per_tumour={r["task"]: _share(x["n_false"], x["n_nonmut"]) for r, x in zip(rs, s)})
        if all("cell" in r["assays"][k] for r in rs):
            c = [r["assays"][k]["cell"] for r in rs]
            cl = {}
            for part in ("all", "in_window", "out_window"):
                src = c if part == "all" else [x[part] for x in c]
                n = sum(x["n_pairs"] for x in src)
                cl[part] = dict(n_pairs=n, share_detected=_share(sum(x["n_detected"] for x in src), n),
                                share_covered=_share(sum(x["n_covered"] for x in src), n),
                                mean_depth=_share(sum(x["sum_depth"] for x in src), n),
                                per_tumour={r["task"]: dict(share_detected=_share(x["n_detected"], x["n_pairs"]),
                                                            mean_depth=_share(x["sum_depth"], x["n_pairs"]))
                                            for r, x in zip(rs, src)})
            cl["unit"] = rs[0]["assays"][k].get("unit", "cell")
            num["cell_level"][k] = cl
    num["site_level_error_aware"], num["specificity_error_aware"] = error_aware(recs)
    for r in recs:
        num["cardelino"][r["task"]] = dict(chance=r["chance"], n_true_clones=r["n_true_clones"],
                                           n_variants=r["cardelino_config"]["n_variants"],
                                           n_variants_dna_config=r.get("cardelino_dna_config", {}).get("n_variants"),
                                           checks=r["checks"],
                                           arms={k: dict({f: v.get(f) for f in (
                                               "balanced_accuracy", "accuracy", "share_prob_above_half",
                                               "share_cells_no_read", "mean_reads_per_cell_variant",
                                               "confident_balanced_accuracy", "failed", "superseded")},
                                               n_iter=v.get("run", {}).get("n_iter"),
                                               min_iter=v.get("run", {}).get("min_iter"),
                                               max_iter=v.get("run", {}).get("max_iter"))
                                               for k, v in r["cardelino"].items()})
    return num


def specificity_note(spec_num):
    """Panel b's one-line account of the calls at non-mutated sites (drawn assays)."""
    ks = [k for k in DRAWN if k in spec_num]
    n_dis = sorted({spec_num[k]["n_nonmut"] for k in ks if k != "visium"})
    n_vis = spec_num["visium"]["n_nonmut"] if "visium" in spec_num else None
    if all(spec_num[k]["n_false"] == 0 for k in ks):
        n = "/".join(str(x) for x in n_dis)
        return f"no non-mutated site called in any assay (n = {n}" + (f"; Visium {n_vis})" if n_vis is not None else ")")
    return "non-mutated sites called: " + ", ".join(f"{LABEL[k]} {spec_num[k]['n_false']}/{spec_num[k]['n_nonmut']}"
                                                     for k in ks)


def draw(recs, num):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle
    import style as S
    S.setup()
    col = _colours()
    fig = S.figure("double", 118)

    # (a) the coverage model -------------------------------------------------------------------------
    S.panel_letter(fig, "a", 2, 3)
    ax = S.ax_mm(fig, 6, 9, 80, 44)
    ax.set_xlim(-0.6, 10.4)
    ax.set_ylim(-2.0, 6.4)
    ax.axis("off")
    L, W = 10.0, 3.9            # transcript drawn 0..L (5'..3'), window the last W (1,000 of ~2,500 b)
    rng = np.random.default_rng(3)
    sites_x = (8.4, 2.6)        # a site in the 3' window, and one outside it
    for y, name, k, full in ((4.4, "10x and Visium, 3′ end", "10x", False),
                             (1.4, "Smart-seq3, full length", "smartseq3", True)):
        ax.text(-0.5, y + 1.25, name, ha="left", va="bottom", fontsize=6.0, color=col[k])
        if not full:
            ax.add_patch(Rectangle((L - W, y - 0.42), W, 0.84, facecolor=S.PASTEL["yellow"], edgecolor="none", zorder=0))
            ax.text(L - W / 2, y - 0.62, "3′ window", ha="center", va="top", fontsize=5.0, color=S.GREY)
        ax.add_patch(FancyBboxPatch((0, y - 0.13), L, 0.26, boxstyle="round,pad=0,rounding_size=0.1",
                                    facecolor="#BDBDBD", edgecolor="none", zorder=1))
        ax.text(L + 0.12, y, "AAAA", ha="left", va="center", fontsize=4.5, color=S.GREY)
        ax.text(-0.1, y, "5′", ha="right", va="center", fontsize=4.5, color=S.GREY)
        lo = L - W if not full else 0.0
        starts = rng.uniform(lo, L - 0.6, size=22 if full else 14)
        for i, x0 in enumerate(starts):
            yy = y + 0.3 + 0.17 * (i % 4)
            ax.plot([x0, x0 + 0.6], [yy, yy], color=col[k], lw=0.9, solid_capstyle="butt", zorder=2)
        for sx in sites_x:
            seen = full or sx >= L - W
            ax.plot([sx], [y], marker="v", ms=4.2, color=S.INK if seen else "white", mec=S.INK, mew=0.6, zorder=3)
            ax.text(sx, y - 0.26, "read" if seen else "not read", ha="center", va="top", fontsize=4.5,
                    color=S.INK if seen else S.GREY)
    # The coverage model's settings (transcript lengths, the 3' window's share of sites, the per-molecule read
    # probability, scDNA's thinning to the reads over the site) are in the caption, not in the figure.

    # (b) site-level detection ------------------------------------------------------------------------
    S.panel_letter(fig, "b", 94, 3)
    site_num, spec_num = num["site_level_error_aware"], num["specificity_error_aware"]
    ax = S.ax_mm(fig, 104, 15, 76, 34)
    xs = np.arange(len(FREQ_LABELS))
    for i, k in enumerate(DRAWN):
        sl = site_num.get(k)
        if sl is None:
            continue
        y = [b["share"] for b in sl["all"]["by_bin"]]
        ls = "--" if k == "smartseq3_10xdepth" else "-"
        ax.plot(xs, y, ls, marker="o", ms=2.2, color=col[k], lw=0.9, label=LABEL[k], zorder=3)
        for b, bb in enumerate(sl["all"]["by_bin"]):
            v = [x for x in bb["per_tumour"].values() if np.isfinite(x)]
            ax.scatter(np.full(len(v), b + (i - 2.5) * 0.06), v, s=1.2, color=col[k], alpha=0.45, zorder=2,
                       linewidths=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(FREQ_LABELS)
    ax.set_xlabel("Site frequency among cancer cells")
    ax.set_ylabel("Share of somatic\nsites called")      # the call rule (alt reads, error floor) is in the caption
    ax.set_ylim(-0.02, 1.13)                       # headroom above 1, kept from the specificity note's layout
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["left"].set_bounds(0, 1)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.01), fontsize=5.0, ncol=3, handlelength=1.6,
              columnspacing=0.8)
    # the calls at non-mutated sites (specificity_note) are stated in the caption, not printed in the figure

    # (c) cell-level detection --------------------------------------------------------------------------
    S.panel_letter(fig, "c", 2, 64)
    ks = [k for k in CELL_DRAWN if k in num["cell_level"]]
    lab = [LABEL[k] + (" spots" if num["cell_level"][k]["unit"] == "spot" else "") for k in ks]
    for j, (metric, xl, logx, x0) in enumerate((("share_detected", "Carrier cell–site pairs\nwith an alt read",
                                                 False, 36),
                                                ("mean_depth", "Mean reads over the site\nper carrier pair", False, 66))):
        ax = S.ax_mm(fig, x0, 70, 22, 38)
        yy = np.arange(len(ks))[::-1]
        val = [num["cell_level"][k]["all"][metric] for k in ks]
        ax.barh(yy, val, color=[col[k] for k in ks], height=0.68)
        for k, y_ in zip(ks, yy):
            if k == "scdna_site":
                continue
            for part, mk in (("in_window", "^"), ("out_window", "v")):
                ax.plot(num["cell_level"][k][part][metric], y_, marker=mk, ms=2.4, color=S.INK, mfc="white",
                        mew=0.5, ls="none", zorder=3)
            for t, pv in num["cell_level"][k]["all"]["per_tumour"].items():
                ax.plot(pv[metric], y_ + 0.25, marker="|", ms=2.0, color=S.INK, alpha=0.5, mew=0.4, ls="none")
        ax.set_yticks(yy)
        ax.set_yticklabels(lab if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        if logx:
            ax.set_xscale("log")
        else:
            ax.set_xlim(0, None)
        ax.set_xlabel(xl)
        S.despine(ax, left=False)
    ax.plot([], [], marker="^", ms=2.4, color=S.INK, mfc="white", mew=0.5, ls="none", label="site in 3′ window")
    ax.plot([], [], marker="v", ms=2.4, color=S.INK, mfc="white", mew=0.5, ls="none", label="site outside")
    ax.legend(loc="lower right", bbox_to_anchor=(1.05, 1.0), fontsize=4.8, ncol=1, handletextpad=0.2)

    # (d) cardelino ------------------------------------------------------------------------------------------
    S.panel_letter(fig, "d", 94, 64)
    ax = S.ax_mm(fig, 102, 70, 78, 38)
    tasks = [r["task"] for r in recs]
    off = {"10x": -0.22, "smartseq3": 0.0, "smartseq3_10xdepth": 0.22}
    for i, t in enumerate(tasks):
        c = num["cardelino"][t]
        ax.plot([i - 0.38, i + 0.38], [c["chance"]] * 2, color=S.GREY, lw=0.8, zorder=1)
        for k, dx in off.items():
            a_ = c["arms"].get(f"true_{k}", {})
            if a_.get("balanced_accuracy") is not None:
                ax.bar(i + dx, a_["balanced_accuracy"], width=0.2, color=col[k], zorder=2,
                       label=LABEL[k] if i == 0 else None)
            d_ = c["arms"].get(f"dna_{k}", {})
            if d_.get("balanced_accuracy") is not None:
                ax.plot(i + dx, d_["balanced_accuracy"], marker="o", ms=2.6, mfc="white", mec=S.INK, mew=0.5,
                        ls="none", zorder=3, label="scDNA-called clones" if (i == 0 and k == "10x") else None)
    ax.plot([], [], color=S.GREY, lw=0.8, label="chance")
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels([f"{_short(t)}\n{num['cardelino'][t]['n_true_clones']} clones" for t in tasks])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("cardelino balanced accuracy")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=5, fontsize=5.0, handlelength=1.2)
    return S.save(fig, "supp_snv")


def make_figure(out_dir, fig4_c=None, write_numbers=True):
    """Draw the figure from the stored per-tumour outputs under ``out_dir`` (summary.json and sites.tsv.gz; nothing
    is simulated) into ``figures/paper1/`` of ``_paths.paper_dir()`` ($ISCC_PAPER_DIR), and write the drawn numbers
    to ``out_dir/supp_snv_numbers.json`` unless ``write_numbers`` is False."""
    recs = collect(out_dir)
    if not recs:
        sys.exit(f"no summaries under {out_dir}")
    num = numbers(recs)
    if fig4_c:                    # Fig 4c's clonealign (copy number) on the same cells, for reference only
        for r in recs:
            p = os.path.join(fig4_c, f"{r['run']}_{r['task']}", "score.json")
            if os.path.exists(p):
                with open(p) as f:
                    fine = json.load(f)["fine"]
                num["cardelino"][r["task"]]["fig4c_clonealign"] = dict(
                    truth_arm=fine["truth_arm"]["balanced_accuracy"], dna_arm=fine["dna_arm"]["balanced_accuracy"],
                    chance=fine["chance"])
    num["scdna_thinning"] = recs[0]["assays"]["scdna_site"]["thinning"]
    path = draw(recs, num)
    num["figure"] = [path, path[:-4] + ".png"]
    print("figure:", path, flush=True)
    if write_numbers:
        with open(os.path.join(out_dir, "supp_snv_numbers.json"), "w") as f:
            json.dump(num, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        print("numbers:", os.path.join(out_dir, "supp_snv_numbers.json"), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "cardelino", "figure"))
    ap.add_argument("--run", default="R11")
    ap.add_argument("--task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", help="scratch for every tool input and output (a job's $SNIC_TMP)")
    ap.add_argument("--fig4-c", help="Fig 4c's per-tumour outputs (OUT/c of fig4_chain.py): the sampling check "
                                     "and the optional scDNA-called clones for cardelino")
    ap.add_argument("--no-numbers", action="store_true",
                    help="figure step: draw only, leave supp_snv_numbers.json as it is (e.g. rendering into the paper)")
    ap.add_argument("--keys", default="", help="cardelino step: the runs to redo, e.g. true_10x,dna_smartseq3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-dna", type=int, default=FC.N_DNA)
    ap.add_argument("--rna-cells", type=int, default=FC.RNA_MAX_CELLS)
    ap.add_argument("--section-max-cells", type=int, default=FC.SECTION_MAX_CELLS)
    ap.add_argument("--cardelino-iter", type=lambda s: tuple(int(x) for x in s.split(",")), default=CARDELINO_ITER,
                    help="min_iter,max_iter of cardelino's Gibbs sampler (default the vignette's 800,1200)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    if a.step == "figure":
        make_figure(a.out, a.fig4_c, write_numbers=not a.no_numbers)
        return
    if not (a.task and a.work):
        sys.exit("run needs --task and --work")
    try:
        (run_tumour if a.step == "run" else rerun_cardelino)(a)
    finally:
        if os.environ.get("SNV_KEEP_WORK") != "1":
            shutil.rmtree(os.path.join(a.work, f"snv_{a.run}_{a.task}"), ignore_errors=True)


if __name__ == "__main__":
    main()
