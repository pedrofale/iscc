"""Synthetic stand-ins for OncoSpot's unreleased reference tables, for timing it only.

OncoSpot's public repository (compbio-mallory/OncoSpot, e471d21) ships the YAML metadata of its
calibrated parameter bundles but not the tables themselves ("A compatible parameter bundle must be
supplied before running an end-to-end simulation", its README), so it cannot run as released. To time
it end to end, this script writes tables with the columns its loaders require and the sizes its
metadata states (genes, reference cells, states, depth quantiles), filled with synthetic values. Run
time depends on those sizes, not on the values, so the timing stands; nothing biological does.

Usage::

    python oncospot_placeholders.py /path/to/OncoSpot [--seed 0]
"""
import argparse
import gzip
import os

import numpy as np
import pandas as pd
import yaml


def visium_depth(root, rng):
    """``depth_histogram.tsv`` and ``reference_spot_umis.tsv.gz`` from the metadata's quantiles: a
    lognormal with the reference's median and its 5-95% spread, cut into the stated number of bins."""
    d = os.path.join(root, "parameters/observation/visium/gse211956_sp4")
    meta = yaml.safe_load(open(os.path.join(d, "metadata.yaml")))
    mu = np.log(meta["median_umis"])
    sigma = (np.log(meta["q95_umis"]) - np.log(meta["q05_umis"])) / (2 * 1.6449)
    umis = np.clip(np.round(rng.lognormal(mu, sigma, meta["n_reference_spots"])),
                   meta["min_umis"], meta["max_umis"]).astype(int)
    edges = np.linspace(umis.min(), umis.max() + 1, meta["histogram_requested_bins"] + 1)
    counts, _ = np.histogram(umis, bins=edges)
    rows = [dict(bin_index=i, lower_umi=int(np.ceil(edges[i])), upper_umi=int(np.floor(edges[i + 1])),
                 count=int(c), probability=c / counts.sum())
            for i, c in enumerate(counts) if c > 0]
    for i, r in enumerate(rows):
        r["bin_index"] = i
    pd.DataFrame(rows).to_csv(os.path.join(d, "depth_histogram.tsv"), sep="\t", index=False)
    with gzip.open(os.path.join(d, "reference_spot_umis.tsv.gz"), "wt") as f:
        pd.DataFrame({"reference_spot_index": np.arange(len(umis)), "total_umis": umis}).to_csv(
            f, sep="\t", index=False)


HALLMARK_STATES = ["PROLIFERATION", "WNT", "MYC", "HYPOXIA", "EMT", "METABOLIC"]
SA1053_STATES = ["G2M", "S_PHASE", "HYPOXIA", "IFN_BROAD", "IFN_INFLAMMATORY", "EMT"]
TME_TYPES = ["T_NK", "B_plasma", "macrophage_myeloid", "fibroblast", "endothelial", "normal_epithelial"]


def _genes(root, n):
    """The first ``n`` autosomal GENCODE genes, protein-coding first, from OncoSpot's own annotation
    (``scripts/setup/prepare_gencode_genes.py`` output), so every gene maps to a CNA bin."""
    g = pd.read_csv(os.path.join(root, "data/annotations/gencode/v50/processed/genes_autosomes.tsv"),
                    sep="\t")
    g = pd.concat([g[g.gene_type == "protein_coding"], g[g.gene_type != "protein_coding"]])
    return g.drop_duplicates("gene_id").iloc[:n][["gene_id", "gene_symbol"]].reset_index(drop=True)


def _fractions(rng, n):
    """Expression fractions over ``n`` genes with a long tail, as in pseudobulk references."""
    x = rng.lognormal(0.0, 2.0, n)
    return x / x.sum()


def tumour_states(root, rng):
    for path, states in (("parameters/HGSOC/hallmark_v1/tumor_states/hgsoc_state_reference.tsv",
                          HALLMARK_STATES),
                         ("parameters/HGSOC/tumor_states/sa1053_state_reference.tsv", SA1053_STATES)):
        n = 3369                                          # SA1053 reference cells (state_metadata.yaml)
        z = rng.standard_normal((n, len(states)))
        df = pd.DataFrame(z, columns=states)
        df.insert(0, "reference_cell_id", [f"ref{i}" for i in range(n)])
        df.to_csv(os.path.join(root, path), sep="\t", index=False)


def malignant(root, rng):
    """The NB2 mean-model parameters, 19,190 genes (expression_metadata.yaml), the SA1053 library
    sizes and the cell-of-origin baseline."""
    for path, states in (("parameters/HGSOC/hallmark_v1/malignant/expression/malignant_expression_parameters.tsv",
                          HALLMARK_STATES),
                         ("parameters/HGSOC/malignant/expression/malignant_expression_parameters.tsv",
                          SA1053_STATES)):
        g = _genes(root, 19190)
        p = _fractions(rng, len(g))
        g["baseline_fraction"] = p
        g["alpha"] = np.log(p)
        g["beta"] = np.where(rng.random(len(g)) < 3756 / 19190, rng.lognormal(np.log(0.42), 0.5, len(g)),
                             0.41535995888056954)
        g["dispersion"] = 0.5689652598822481
        g["state_effect_lower"] = -1.0
        g["state_effect_upper"] = 1.0
        calibrated = rng.random(len(g)) < 5143 / 19190
        for s in states:
            g[f"gamma_{s}"] = np.where(calibrated, np.clip(rng.normal(0, 0.1, len(g)), -0.3, 0.3), 0.0)
        g["beta_source"], g["beta_replaced"], g["dispersion_source"] = "placeholder", False, "pooled"
        g.to_csv(os.path.join(root, path), sep="\t", index=False)
    pd.DataFrame({"library_size": np.round(rng.lognormal(np.log(8000), 0.6, 3369)).astype(int) + 500}).to_csv(
        os.path.join(root, "parameters/HGSOC/malignant/expression/sa1053_library_sizes.tsv"), sep="\t",
        index=False)
    g = _genes(root, 21184)
    p = _fractions(rng, len(g))
    g["baseline_fraction_donor_balanced"] = p
    g["alpha_origin"] = np.log(p)
    g.to_csv(os.path.join(root, "parameters/HGSOC/malignant/cell_of_origin/early_secretory_ftsec_baseline.tsv"),
             sep="\t", index=False)


def nonmalignant(root, rng):
    """Six normal-type baselines over 22,427 genes and per-type library sizes, from
    tme_expression_metadata.yaml and library_size_metadata.yaml."""
    d = os.path.join(root, "parameters/HGSOC/nonmalignant/expression")
    g = _genes(root, 22427)
    rows = []
    for t in TME_TYPES:
        p = _fractions(rng, len(g))
        rows.append(pd.DataFrame({"gene_id": g.gene_id, "gene_symbol": g.gene_symbol, "broad_cell_type": t,
                                  "baseline_fraction": p, "alpha": np.log(p)}))
    pd.concat(rows).to_csv(os.path.join(d, "tme_expression_baselines.tsv"), sep="\t", index=False)
    lib = yaml.safe_load(open(os.path.join(d, "library_size_metadata.yaml")))["summary_by_cell_type"]
    rows = []
    for t in TME_TYPES:
        s = lib[t]
        sigma = (np.log(s["q95"]) - np.log(s["q05"])) / (2 * 1.6449)
        x = np.clip(np.round(rng.lognormal(np.log(s["median"]), sigma, s["n_cells"])), s["min"], s["max"])
        rows.append(pd.DataFrame({"broad_cell_type": t, "library_size": x.astype(int),
                                  "reference_cell_id": [f"{t}_{i}" for i in range(len(x))]}))
    pd.concat(rows).to_csv(os.path.join(d, "library_sizes_by_cell_type.tsv"), sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    visium_depth(a.root, rng)
    tumour_states(a.root, rng)
    malignant(a.root, rng)
    nonmalignant(a.root, rng)


if __name__ == "__main__":
    main()
