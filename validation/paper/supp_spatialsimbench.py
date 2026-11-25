"""S Fig: iscc on SpatialSimBench's scale (Liang et al. 2025, Genome Biology).

SpatialSimBench scores spatial simulators on ten real datasets and deposits the KDE statistic of every
metric for 13 simulator configurations (figshare 10.6084/m9.figshare.26054188, v4). Its BREAST dataset is
a Visium section of a breast cancer from Wu et al. 2021: 4,744 spots on its 1,000 most spatially variable
genes. This script puts iscc on that scale for the spot- and gene-level metrics: it measures sections of
an iscc tumour with the paper regime's Visium settings, keeps the reference's genes, subsamples to the
reference's number of spots (the KDE statistic grows with the sample size), and computes the metrics
with ``tools/spatialsimbench_ks.R`` (the OpenProblems port of SpatialSimBench's code).

The Visium assay holds dense spots x genes tables over the 36,601 human genes, so a section of about
5,000 spots does not fit a laptop; the tumour is sectioned in four windows of 70 x 70 demes, each
measured separately, and their spots pooled.

    python validation/paper/supp_spatialsimbench.py section --tumour pregrown:immune
    python validation/paper/supp_spatialsimbench.py metrics --tumour pregrown:immune
    python validation/paper/supp_spatialsimbench.py section --tumour pregrown:immune --variant fitted   # after published
    python validation/paper/supp_spatialsimbench.py metrics --tumour pregrown:immune --variant fitted
    python validation/paper/supp_spatialsimbench.py figure --tumour pregrown:immune   # S15 Fig, both variants

S15 Fig uses ``pregrown:immune``, the tumour of the paper's regime with immune cells in the stroma: the
real section is a breast cancer with them, and the immunoglobulins of its plasma cells carry most of
its gene variance, which ``pregrown:ductal`` (no immune cells) cannot hold.

The tumour must be on the chromosome-arm genome (the human genes come from ``HumanGenomeLayer``): the
pregrown tumours are, the registry's 12-segment runs (R0 to R13) are not.

``ISCC_SIMBENCH_DIR`` holds the reference (``ref/breast_counts.mtx``, ``ref/breast_genes.txt``) and the
published table (``data/spatialsimbenchV1.csv``); outputs go to ``<ISCC_SIMBENCH_DIR>/iscc/<tumour>/``.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import regime as RG  # noqa: E402
import registry as REG  # noqa: E402

BENCH = os.environ.get("ISCC_SIMBENCH_DIR", os.path.expanduser("~/projects/iscc-workspace/runs/simbench"))
RSCRIPT = os.environ.get("ISCC_SIMBENCH_RSCRIPT", os.path.expanduser("~/miniconda3/envs/iscc-simbench/bin/Rscript"))
WINDOW, SEED = 70, 1


def _load(tumour):
    if tumour.startswith("pregrown:"):
        import iscc.pregrown
        return getattr(iscc.pregrown, tumour.split(":", 1)[1])()
    run, task = tumour.split("/")
    return REG.load(run, task)


def _out(tumour, variant="published"):
    d = os.path.join(BENCH, "iscc", tumour.replace("/", "_").replace(":", "_"), variant)
    os.makedirs(d, exist_ok=True)
    return d


def _windows(t):
    """Four windows of WINDOW x WINDOW demes, centred in the four quadrants of the primary field."""
    q = t.grid_size // 4
    return [t.primary_window(side=WINDOW, center=(r, c)) for r in (q, 3 * q) for c in (q, 3 * q)]


def _reference_genes(layer_genes):
    """The reference's 1,000 genes as human-layer columns. R turned '-' into '.' in the names, so a name
    without an exact match is tried with every '.' as '-'."""
    ref = [l.strip() for l in open(os.path.join(BENCH, "ref", "breast_genes.txt")) if l.strip()]
    by_name = layer_genes.reset_index().drop_duplicates("gene_name").set_index("gene_name")["gene_id"]
    cols = []
    for g in ref:
        cols.append(by_name.get(g, by_name.get(g.replace(".", "-"))))
    return ref, cols


def _reference_median():
    m = scipy.io.mmread(os.path.join(BENCH, "ref", "breast_counts.mtx")).tocsr()
    return float(np.median(np.asarray(m.sum(axis=1)).ravel()))


def _reference_fit():
    """``estimate_visium`` on the reference section: the spread of the spot depth (``sigma_counts``)
    and the capture field (``field_sigma``, ``field_lengthscale``), with the length scale moved from
    the reference's spot pitch to iscc's. The depth itself is matched separately, and the count
    overdispersion is left at iscc's: fitted on 1,000 spatially variable genes it holds their biology."""
    from iscc.data import Visium, estimate_visium
    from iscc.data.estimate_visium import _pitch
    counts = scipy.io.mmread(os.path.join(BENCH, "ref", "breast_counts.mtx")).toarray()
    xy = pd.read_csv(os.path.join(BENCH, "ref", "breast_spots.csv"))[["row", "col"]].to_numpy(float)
    h = estimate_visium(counts, coords=xy).hypers
    pitch = Visium().hypers.spot_pitch
    return dict(sigma_counts=h.sigma_counts, field_sigma=h.field_sigma,
                field_lengthscale=h.field_lengthscale / _pitch(xy) * pitch)


def section(tumour, variant="published"):
    """``published``: the paper regime's Visium as it stands. ``depth``: the same, with the depth
    (``mu_counts``) scaled so that the median count per spot on the reference's genes matches the
    reference's, as the simulators SpatialSimBench scores are fitted to the reference. ``fitted``:
    the depth matched in the same way, and the spread of the depth and the capture field fitted to
    the reference by ``estimate_visium`` (``_reference_fit``)."""
    from iscc.data import Visium
    from iscc.data.human import HumanGenomeLayer
    from iscc.sample import Resection
    t = _load(tumour)
    layer = HumanGenomeLayer(t)
    expression = layer.expression(t)
    ref, cols = _reference_genes(layer.genes.set_index("gene_id"))
    found = [c for c in cols if c is not None]
    settings = dict(RG.VISIUM)
    if variant in ("depth", "fitted"):
        published = pd.read_csv(os.path.join(_out(tumour), "section.csv"), index_col=0).iloc[:, 0]
        scale = _reference_median() / float(published["median_counts_per_spot"])
        if variant == "fitted":
            fit = _reference_fit()
            h = Visium().hypers
            # the depth and the field are mean-1 lognormals, so their median is exp(-sd^2 / 2) of the
            # mean: a narrower spread needs a lower mean for the same median
            scale *= np.exp(0.5 * (fit["sigma_counts"] ** 2 + fit["field_sigma"] ** 2
                                   - h.sigma_counts ** 2 - h.field_sigma ** 2))
            settings.update(fit)
            print("fitted to the reference: " + ", ".join(f"{k} {v:.3f}" for k, v in fit.items()), flush=True)
        settings["mu_counts"] = Visium().hypers.mu_counts * scale
        print(f"depth matched to the reference: mu_counts x {scale:.3f} = {settings['mu_counts']:.0f}", flush=True)
    parts = []
    specimen = Resection(t)
    for k, window in enumerate(_windows(t)):
        sec = specimen.slice(window, depth_frac=0.5, max_cells=10 ** 7, layers="visium")
        vis = Visium(seed=SEED + k, section_frac=1.0, **settings).run(sec, grid_side=WINDOW, gene_lift=expression)
        in_tissue = (vis.obs["n_cells"] > 0).to_numpy()
        counts = vis.spot_counts.loc[in_tissue, found]
        parts.append(counts.reset_index(drop=True))
        print(f"window {k}: {int(in_tissue.sum())} spots in the tissue", flush=True)
        del vis, sec
    pooled = pd.concat(parts, ignore_index=True)
    n_ref = scipy.io.mminfo(os.path.join(BENCH, "ref", "breast_counts.mtx"))[0]
    keep = np.sort(np.random.default_rng(SEED).choice(len(pooled), size=min(n_ref, len(pooled)), replace=False))
    full = np.zeros((len(keep), len(ref)), dtype=np.int64)       # a reference gene iscc lacks stays at 0
    pos = {c: i for i, c in enumerate(cols) if c is not None}
    for c in found:
        full[:, pos[c]] = pooled[c].to_numpy()[keep]
    out = _out(tumour, variant)
    scipy.io.mmwrite(os.path.join(out, "counts.mtx"), sp.csr_matrix(full))
    pd.Series({"spots_pooled": len(pooled), "spots_kept": len(keep), "reference_genes": len(ref),
               "genes_matched": len(found), "genes_all_zero": int((full.sum(axis=0) == 0).sum()),
               "median_counts_per_spot": float(np.median(full.sum(axis=1)))}).to_csv(os.path.join(out, "section.csv"))
    print(open(os.path.join(out, "section.csv")).read())


def metrics(tumour, variant="published"):
    out = _out(tumour, variant)
    subprocess.run([RSCRIPT, os.path.join(HERE, "tools", "spatialsimbench_ks.R"),
                    os.path.join(BENCH, "ref", "breast_counts.mtx"), os.path.join(out, "counts.mtx"),
                    os.path.join(out, "ks.csv")], check=True)
    ours = pd.read_csv(os.path.join(out, "ks.csv"))
    pub = pd.read_csv(os.path.join(BENCH, "data", "spatialsimbenchV1.csv"))
    pub = pub[(pub.datasetID == "BREAST") & (pub.metric == "KDEstat")].copy()
    # the published table has no level column: per method, the first of two rows of a metric is the
    # gene-level one and the second the spot-level one (MeanVar and MeanFrac are gene-level only, TMM,
    # libsize, libFracZero and EffLibsize spot-level only)
    pub["level"] = np.where(pub.groupby(["method", "auxData"]).cumcount() == 0, "gene", "spot")
    pub.loc[pub.auxData.isin(["TMM", "libsize", "libFracZero", "EffLibsize"]), "level"] = "spot"
    pub = pub[pub.auxData.isin(ours.metric)]
    table = pub.pivot_table(index=["level", "auxData"], columns="method", values="result")
    table["iscc"] = ours.set_index(["level", "metric"])["zstat"].reindex(table.index).to_numpy()
    table.to_csv(os.path.join(out, "scale.csv"))
    print(table.round(2).to_string())


METRIC_LABEL = {"scaledVar": "scaled variance", "scaledMean": "scaled mean", "pearson": "correlation",
                "MeanVar": "mean and variance", "MeanFrac": "mean and zero fraction", "fracZero": "zero fraction",
                "TMM": "TMM factor", "libsize": "library size", "libFracZero": "library size and zeros",
                "EffLibsize": "effective library size"}


def figure(tumour, variant=None):
    """S15 Fig: the KDE z-statistic of every gene- and spot-level metric on SpatialSimBench's breast
    section, for its 13 published simulator configurations and for iscc as published and with its spot
    depth fitted to the section (lower is closer to the real section)."""
    import style as S
    tables = {v: pd.read_csv(os.path.join(_out(tumour, v), "scale.csv"), index_col=[0, 1])
              for v in ("published", "fitted") if os.path.exists(os.path.join(_out(tumour, v), "scale.csv"))}
    pub = next(iter(tables.values())).drop(columns="iscc")
    S.setup()
    fig = S.figure(165, 52)
    out = {}
    for k, level in enumerate(("gene", "spot")):
        rows = [m for m in pub.loc[level].index]
        ax = S.ax_mm(fig, 30 + k * 82, 6, 48, 4.2 * len(rows))
        for i, m in enumerate(rows):
            others = pub.loc[(level, m)].to_numpy(float)
            ax.scatter(others, np.full(len(others), i), s=6, color=S.SOURCE["real"], zorder=2, linewidths=0)
            for v, colour, marker in (("published", S.SOURCE["iscc"], "o"), ("fitted", S.INK, "D")):
                if v in tables:
                    z = float(tables[v].loc[(level, m), "iscc"])
                    ax.scatter([z], [i], s=12, color=colour, marker=marker, zorder=3, linewidths=0)
                    rank = int((others < z).sum()) + 1
                    out.setdefault(v, {})[f"{level}/{m}"] = dict(z=round(z, 2), rank=rank, of=len(others) + 1)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([METRIC_LABEL.get(m, m) for m in rows], fontsize=5.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlabel("KDE z-statistic", fontsize=5.5)
        ax.set_title(f"{level}-level metrics", fontsize=6)
        ax.tick_params(labelsize=5.5)
        S.despine(ax)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", ms=2.5, color=S.SOURCE["real"], label="13 published configurations"),
               Line2D([], [], ls="", marker="o", ms=3.2, color=S.SOURCE["iscc"], label="iscc"),
               Line2D([], [], ls="", marker="D", ms=3.0, color=S.INK, label="iscc, depth fitted to the section")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=5.5, frameon=False)
    path = S.save(fig, "supp_simbench")
    with open(os.path.join(os.path.dirname(_out(tumour, "published")), "figure_numbers.json"), "w") as f:
        import json
        json.dump(out, f, indent=1)
    print(path)
    for v, d in out.items():
        print(v, {k: (r["z"], f'{r["rank"]}/{r["of"]}') for k, r in d.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["section", "metrics", "figure"])
    ap.add_argument("--tumour", required=True, help="pregrown:<name> or <run>/<task> of the registry")
    ap.add_argument("--variant", choices=["published", "depth", "fitted"], default="published",
                    help="the Visium as published, with its depth matched to the reference, or with its "
                         "depth and its spread fitted to the reference")
    a = ap.parse_args()
    {"section": section, "metrics": metrics, "figure": figure}[a.cmd](a.tumour, a.variant)


if __name__ == "__main__":
    main()
