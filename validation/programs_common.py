"""Shared data generation + scoring for the gene-program recovery benchmark (scDEF vs cNMF).

`DESIGN_expression.md` §4.2. The question: can a program-inference tool recover the TRUE gene
programs from iscc's scRNA counts, and **how does that degrade as SNV/CNA burden rises?** iscc can
ask this because it knows the true `loading` matrix and the true per-cell `z` (`program_truth`) —
ground truth no real dataset provides.

**The hypothesis under test (§4.2).** CNAs are CONTIGUOUS, so a high CNA burden induces *positional*
co-expression: genes co-vary because they share a copy-number segment, not because they share a
function. A factor model can absorb that as **spurious "programs"**, so true-program recovery should
degrade with fraction-genome-altered. The diagnostic that separates artefact from biology is the
orthogonality itself — are a factor's genes **positionally clustered** (a CNA artefact) or
**scattered** (a real program)? That is what `positional_clustering` below measures.

This is a HYPOTHESIS, not a foregone conclusion. If CNA burden does not degrade recovery, that is a
reportable — and reassuring — result about the tools, and it must be reported as such rather than
tuned into existence.

Each external tool runs in its own dedicated conda env per `README_integration.md`
(`iscc-scdef`, `iscc-cnmf`); this module stays in the core `iscc` env and never imports them.
"""
import os
import subprocess
import tempfile

import numpy as np
from scipy.optimize import linear_sum_assignment

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))

SCDEF_PYTHON = os.environ.get("ISCC_SCDEF_PYTHON",
                              os.path.join(HOME, "miniconda3/envs/iscc-scdef/bin/python"))
CNMF_PYTHON = os.environ.get("ISCC_CNMF_PYTHON",
                             os.path.join(HOME, "miniconda3/envs/iscc-cnmf/bin/python"))


def scdef_available():
    return os.path.exists(SCDEF_PYTHON)


def cnmf_available():
    return os.path.exists(CNMF_PYTHON)


# A tumour with room for several segmental CNAs and enough cells for a factor model to fit. The
# program layer is ON with defaults that keep within-clone heterogeneity healthy (so `clone == state`
# never trivialises the benchmark — see tumor.diagnose()'s `clone_is_state` check).
GENOME = {"n_segments": 10, "segment_size": 40}          # 400 genes over 10 "chromosomes"
SELECTION = {"prop_driver": 0.2, "prop_dispersal": 0.0, "prop_immune_resistance": 0.0,
             "prop_treatment_resistance": 0.0}
DEME = {"carrying_capacity": 8, "initial_cancer_cells": 8}
SPATIAL = {"grid_size": 16, "structure_radius": 0}       # cancer only: programs, not cell types
CANCER = {"division_rate": 0.6, "death_rate": 0.03, "max_birth_rate": 0.98,
          "mutation_rate": 1.0, "dispersal_rate": 0.4}

N_PROGRAMS = 6
PROGRAM_PARAMS = {"n_programs": N_PROGRAMS, "n_genes_per_program": 25, "program_overlap": 0.1,
                  "loading_strength": {"mean": 1.0, "sd": 0.3}, "loading_sparsity": 1.0,
                  "program_genomic_scatter": 1.0}
ACTIVITY_PARAMS = {"n_active_programs_per_cell": 3, "activity_dist": "lognormal",
                   "activity_mean": 1.0, "activity_sd": 0.5, "activity_noise": 0.2}
COUPLING_PARAMS = {"phenotype_program_strength": 0.5}


def expression_params(scatter=1.0):
    return {"program_params": dict(PROGRAM_PARAMS, program_genomic_scatter=scatter),
            "activity_params": ACTIVITY_PARAMS,
            "coupling_params": COUPLING_PARAMS,
            "dosage_params": {"dosage_sensitivity_mean": 0.7, "dosage_sensitivity_sd": 0.25,
                              "dosage_saturation": 8, "allele_specific": False},
            "snv_effect_params": {"p_lof": 0.1, "p_missense": 0.3, "p_splice": 0.05,
                                  "p_silent": 0.55, "nmd_strength": 0.2,
                                  "snv_expression_effect": 0.5}}


def grow_tumor(seed=1, steps=12000, mutation_rate=None, cnv_prob=None, amp_prob=None,
               scatter=1.0, n_snvs_per_allele=None):
    """Grow one tumour at a given SNV/CNA burden with the program layer on."""
    from iscc.tumor.models import GenotypeTumor
    cancer = dict(CANCER)
    if mutation_rate is not None:
        cancer["mutation_rate"] = mutation_rate
    if n_snvs_per_allele is not None:
        cancer["n_snvs_per_allele"] = n_snvs_per_allele
    if cnv_prob is not None:
        cancer["cnv_prob"] = cnv_prob
        cancer["snv_prob"] = 1.0 - cnv_prob
    if amp_prob is not None:
        cancer["amp_prob"] = amp_prob
    t = GenotypeTumor(seed=seed, genome_params=GENOME, selection_params=SELECTION,
                      cancer_cell_params=cancer, deme_params=DEME, spatial_params=SPATIAL,
                      expression_params=expression_params(scatter))
    t.grow(n_steps=steps, seed=seed)
    return t


def counts_anndata(tumor, seed=0, max_cells=600, protocol="10x"):
    """iscc expression -> a raw-count AnnData, through iscc's OWN scRNA assay.

    Uses :class:`iscc.data.scRNA` rather than a hand-rolled draw, so the counts carry the assay
    layer's variable library size, negative-binomial overdispersion, ambient soup and dropout — the
    same path `clonealign`'s and RCTD's datasets take.

    This used to normalise every cell's expression to a FIXED depth and draw Poisson, which made
    library size effectively constant: measured on the shipped cohort, CV 0.69% and a 1.04x spread
    between the smallest and largest cell, against 50-100% in real 10x data. The docstring justified
    it as "the benchmark is about whether the PROGRAM structure survives, not about UMI realism" —
    defensible for an internal script, not for a dataset a published notebook analyses, and
    inconsistent with every other tool dataset. Program recovery is HARDER under a realistic read
    layer, which is the point: the benchmark should run on the data iscc actually claims to produce.
    """
    import anndata as ad

    from iscc.data import scRNA

    cd = tumor.cell_data
    is_cancer = np.array([tumor.genotypes[g].type == "cancer"
                          for g in cd["cell_type"]["cell_id"].values])
    cancer_cells = list(np.asarray(cd["cell_exp"].index)[is_cancer])
    rng = np.random.default_rng(seed)
    if len(cancer_cells) > max_cells:
        pick = np.sort(rng.choice(len(cancer_cells), size=max_cells, replace=False))
        cancer_cells = [cancer_cells[i] for i in pick]

    rna = scRNA(n_cells=len(cancer_cells), protocol=protocol, seed=seed).run(
        cd, cell_subset=cancer_cells)
    counts = rna.observed_counts

    # Align the program ground truth to the cells the assay actually returned, by NAME -- the assay
    # may drop or reorder cells, and a positional zip would silently mispair truth with counts.
    Z = np.asarray(cd["cell_program"].reindex(counts.index).values, dtype=float)

    a = ad.AnnData(counts.values.astype(np.float32))
    a.var_names = list(counts.columns)
    a.obs_names = [f"C{i}" for i in range(counts.shape[0])]
    a.layers["counts"] = a.X.copy()
    return a, Z


def run_tool(tool, adata, k, seed=0, n_epoch=300, n_iter=20, batch_key=None):
    """Run scDEF or cNMF in its own env; returns the loadings/activities dict, or None if absent.

    `batch_key` (scDEF only) names an `adata.obs` column and switches on scDEF's OWN batch-correction
    path — the honest way to test correction, rather than pre-correcting the counts by hand.
    """
    py = SCDEF_PYTHON if tool == "scdef" else CNMF_PYTHON
    if not os.path.exists(py):
        return None
    runner = os.path.join(HERE, f"{tool}_runner.py")
    with tempfile.TemporaryDirectory() as d:
        in_h5ad, out_npz = os.path.join(d, "in.h5ad"), os.path.join(d, "out.npz")
        adata.write_h5ad(in_h5ad)
        extra = str(n_epoch) if tool == "scdef" else str(n_iter)
        cmd = [py, runner, in_h5ad, out_npz, str(k), extra, str(seed)]
        if tool == "scdef":
            cmd.append(batch_key or "none")
        elif batch_key is not None:
            raise ValueError("cNMF has no batch-correction path; batch_key is scDEF-only")
        # Sanitise the child's environment. A Jupyter kernel exports
        # MPLBACKEND=module://matplotlib_inline.backend_inline, subprocess inherits os.environ, and
        # the tool env's matplotlib has no matplotlib_inline — so `import scdef` dies at import time
        # with "Key backend: ... is not a valid value". The tool then works from a shell and fails
        # from a notebook, which is how this hid behind a silent NMF fallback. PYTHONPATH is dropped
        # for the same reason: it would let the PARENT env's packages shadow the tool env's.
        env = {k: v for k, v in os.environ.items() if k not in ("MPLBACKEND", "PYTHONPATH")}
        env["MPLBACKEND"] = "Agg"          # headless, always importable
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0 or not os.path.exists(out_npz):
            raise RuntimeError(f"{tool} runner failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        d_npz = np.load(out_npz, allow_pickle=True)
        return {k_: d_npz[k_] for k_ in d_npz.files}


# ------------------------------------------------------------------ scoring
def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))


def positional_clustering(weights, segment_sizes, top_n=25):
    """THE diagnostic (§4.2): is a factor's gene set positionally CLUSTERED or SCATTERED?

    Takes the factor's top-loaded genes and returns the largest fraction of them falling in any one
    genomic segment. A real (functional) program is scattered, so this sits near 1/n_segments; a
    factor that has absorbed a copy-number segment is contiguous, so it approaches 1.0. This is what
    separates "the tool found a program" from "the tool found a CNA and called it a program".
    """
    offs = np.concatenate([[0], np.cumsum(segment_sizes)]).astype(int)
    top = np.argsort(weights)[::-1][:top_n]
    seg_of = np.searchsorted(offs, top, side="right") - 1
    counts = np.bincount(seg_of, minlength=len(segment_sizes))
    return float(counts.max() / max(len(top), 1))


def scattered_null(segment_sizes, top_n=25, n_draw=2000, seed=0):
    """The null for `positional_clustering`: what does a genuinely SCATTERED gene set score?

    Not 1/n_segments — that is the expected fraction PER segment, whereas the statistic is the MAX
    over segments, which is larger by chance alone. Estimate it by drawing random gene sets, so panel
    D compares against the right line rather than an appealing but wrong one.
    """
    rng = np.random.default_rng(seed)
    n_genes = int(np.sum(segment_sizes))
    offs = np.concatenate([[0], np.cumsum(segment_sizes)]).astype(int)
    out = np.empty(n_draw)
    for i in range(n_draw):
        top = rng.choice(n_genes, size=min(top_n, n_genes), replace=False)
        seg_of = np.searchsorted(offs, top, side="right") - 1
        out[i] = np.bincount(seg_of, minlength=len(segment_sizes)).max() / len(top)
    return float(out.mean()), float(np.quantile(out, 0.95))


def score_recovery(true_loading, true_z, inferred, segment_sizes, top_n=25, spurious_cosine=0.3):
    """Hungarian-match each TRUE program to its best inferred factor and score the match.

    Returns per-true-program loading cosine + gene-set Jaccard + activity correlation, the count of
    SPURIOUS factors, and the positional-clustering diagnostic split by matched vs spurious.

    **On the definition of "spurious."** It is NOT "the factors the Hungarian assignment left over":
    that count is just `K_inferred - K_true` by construction (and is identically 0 whenever the tool
    is given exactly as many factors as there are true programs), which measures nothing. A factor is
    spurious when it resembles **no true program at all** — max cosine over every true program below
    `spurious_cosine` — so the count is a real property of the fit. Give the tool MORE factors than
    truth (see `validate_programs.py --k-factors`) or it has no room to invent one.
    """
    W, A = np.asarray(inferred["loadings"]), np.asarray(inferred["activities"])
    K_true, K_inf = true_loading.shape[0], W.shape[0]

    C = np.zeros((K_true, K_inf))
    for i in range(K_true):
        for j in range(K_inf):
            C[i, j] = _cosine(true_loading[i], W[j])
    rows, cols = linear_sum_assignment(-C)

    cosines, jaccards, acts = [], [], []
    for i, j in zip(rows, cols):
        cosines.append(C[i, j])
        true_set = set(np.argsort(true_loading[i])[::-1][:top_n])
        inf_set = set(np.argsort(W[j])[::-1][:top_n])
        union = true_set | inf_set
        jaccards.append(len(true_set & inf_set) / len(union) if union else 0.0)
        aj = A[:, j]
        acts.append(0.0 if aj.std() == 0 or true_z[:, i].std() == 0
                    else float(np.corrcoef(true_z[:, i], aj)[0, 1]))

    # spurious = resembles no true program (see docstring), regardless of the assignment
    best_per_factor = C.max(axis=0) if K_true else np.zeros(K_inf)
    spurious = [j for j in range(K_inf) if best_per_factor[j] < spurious_cosine]
    real = [j for j in range(K_inf) if j not in spurious]
    clust_matched = [positional_clustering(W[j], segment_sizes, top_n) for j in real]
    clust_spurious = [positional_clustering(W[j], segment_sizes, top_n) for j in spurious]

    return dict(
        loading_cosine=float(np.mean(cosines)), gene_jaccard=float(np.mean(jaccards)),
        activity_corr=float(np.mean(acts)), n_spurious=len(spurious), n_inferred=K_inf,
        clustering_matched=float(np.mean(clust_matched)) if clust_matched else float("nan"),
        clustering_spurious=float(np.mean(clust_spurious)) if clust_spurious else float("nan"),
        per_program_cosine=np.array(cosines), per_program_activity=np.array(acts),
    )


def fga(tumor):
    """Fraction of the genome altered (copy number != 2), averaged over cancer cells."""
    cd = tumor.cell_data
    is_cancer = np.array([tumor.genotypes[g].type == "cancer"
                          for g in cd["cell_type"]["cell_id"].values])
    cnv = cd["cell_cnv"].values[is_cancer]
    return float((cnv != 2).mean()) if cnv.size else float("nan")


def snv_burden(tumor):
    """Mean fraction of sites mutated per cancer cell."""
    cd = tumor.cell_data
    is_cancer = np.array([tumor.genotypes[g].type == "cancer"
                          for g in cd["cell_type"]["cell_id"].values])
    snv = cd["cell_snv"].values[is_cancer]
    return float((snv > 0).mean()) if snv.size else float("nan")
