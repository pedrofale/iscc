"""Supplementary figure: every measurement bias of iscc, by step and technology.

A grid with one row per step (sampling, library, sequencing, spatial, batch) and one column per
technology (WGS, scDNA, scRNA, Visium). Each cell lists its biases as ``bias: quantity default`` in
plain words (:data:`PLAIN`; the figure prints no parameter names), with a mark for how the default is
set: fitted from real data by ``estimate_dna`` / ``estimate_rna`` / ``estimate_visium``, fitted only
when the real data carry what the fit needs (the condition is printed, :data:`FITTED_GIVEN`), or set by
the user (a prior or preset that no estimator touches).

The facts live in :data:`BIASES`, built from the code: every entry names where its default lives
(``src``) and, when an estimator fits it, the line that records the fit (``fit_src``), as
``path:line`` relative to the repository root. ``python validation/paper/supp_biases.py --check``
confirms that each of those lines still mentions the parameter, so a refactor that moves a default
shows up here. Fig 1c's strip reads the same structure (:func:`biases`).

Run from the iscc repository root:
    python validation/paper/supp_biases.py --check     # verify every path:line
    python validation/paper/supp_biases.py             # draw figures/paper1/supp_biases.pdf
"""
import os
import sys
import textwrap
from collections import namedtuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

STEPS = ("sampling", "library", "sequencing", "spatial", "batch")
TECHS = ("WGS", "scDNA", "scRNA", "Visium")
FIT, COND, SET = "fitted", "conditional", "set"

Bias = namedtuple("Bias", "step techs bias param default how cond src fit_src")
_D, _B, _R, _V = "src/iscc/data/dna.py", "src/iscc/data/batch.py", "src/iscc/data/rna.py", "src/iscc/data/visium.py"
_ED, _ER, _EV = "src/iscc/data/estimate_dna.py", "src/iscc/data/estimate.py", "src/iscc/data/estimate_visium.py"
_DIS, _RES = "src/iscc/sample/dissociation/dissociation.py", "src/iscc/sample/resection/resection.py"

# One entry per (bias, technologies that share its default and fit status). ``src`` / ``fit_src`` are
# tuples of "path:line"; the default and the estimator's bookkeeping must be on those lines.
BIASES = [
    # --- sampling: the sample-prep biology, before any library is made -------------------------------
    Bias("sampling", ("scDNA", "scRNA"), "cell-type recovery", "recovery_probs",
         "immune 0.4 ... cancer 0.9", SET, None,
         (f"{_DIS}:34",), None),                                       # DEFAULT_RECOVERY
    Bias("sampling", ("scRNA",), "dissociation stress", "stress_strength", "0 (random 5% of genes)", SET, None,
         (f"{_DIS}:71", f"{_DIS}:117"), None),
    Bias("sampling", ("Visium",), "section thickness", "depth_frac", "0.5", SET, None,
         (f"{_RES}:192",), None),
    # --- library: capture and amplification ------------------------------------------------------------
    Bias("library", ("WGS", "scDNA"), "GC bias", "gc_curve_sigma", "0.20 (panel 0.05)", COND, "needs GC",
         (f"{_D}:41", f"{_D}:47"), (f"{_ED}:449",)),
    Bias("library", ("WGS", "scDNA"), "capture variability", "capture_sigma", "0 (panel 0.45)", COND,
         "WES or panel", (f"{_D}:41", f"{_D}:47"), (f"{_ED}:475",)),
    Bias("library", ("WGS", "scDNA"), "FFPE C>T damage", "ffpe_ct_rate", "0", COND, "needs C>T mask",
         (f"{_B}:709",), (f"{_ED}:492",)),
    Bias("library", ("scDNA",), "amplification lumpiness", "kappa", "5", FIT, None,
         (f"{_D}:530",), (f"{_ED}:470",)),
    Bias("library", ("scDNA",), "allelic dropout", "ado_rate", "0.20", COND, "needs het sites",
         (f"{_D}:530",), (f"{_ED}:500",)),
    Bias("library", ("scDNA",), "allelic imbalance", "beta_binom_conc", "30", COND, "needs het sites",
         (f"{_D}:531",), (f"{_ED}:503",)),
    Bias("library", ("scDNA",), "doublets", "doublet_rate", "0.02", SET, None,
         (f"{_D}:531",), (f"{_ED}:66",)),                              # estimate_dna _PRIOR_ONLY
    Bias("library", ("scRNA",), "count overdispersion", "dispersion", "0.30", FIT, None,
         (f"{_R}:40",), (f"{_ER}:324",)),
    Bias("library", ("scRNA",), "dropout", "dropout_mid", "1.5 (Smart-seq3 0)", COND, "fit_dropout=True",
         (f"{_R}:41", f"{_R}:52"), (f"{_ER}:330",)),
    Bias("library", ("scRNA",), "ambient RNA", "ambient_frac", "0.05", SET, None,
         (f"{_R}:41",), (f"{_ER}:60",)),                               # estimate_rna _PRIOR_ONLY
    Bias("library", ("scRNA",), "doublets", "doublet_rate", "0.05", SET, None,
         (f"{_R}:41",), (f"{_ER}:60",)),
    Bias("library", ("Visium",), "count overdispersion", "kappa", "50", FIT, None,
         (f"{_B}:930",), (f"{_EV}:631",)),
    Bias("library", ("Visium",), "ambient RNA", "ambient_frac", "0.05", SET, None,
         (f"{_B}:928",), (f"{_EV}:64",)),                              # estimate_visium _PRIOR_ONLY
    # --- sequencing: depth and its dispersion -----------------------------------------------------------
    Bias("sequencing", ("WGS", "scDNA"), "depth", "mu_depth", "30 (x0.3 single cell)", FIT, None,
         (f"{_D}:41", f"{_D}:270"), (f"{_ED}:440",)),
    Bias("sequencing", ("WGS",), "depth dispersion", "kappa", "2000", FIT, None,
         (f"{_D}:360",), (f"{_ED}:470",)),
    Bias("sequencing", ("WGS", "scDNA"), "base errors", "error_rate", "0.001", COND, "needs variant mask",
         (f"{_B}:704",), (f"{_ED}:488",)),
    Bias("sequencing", ("scRNA",), "library size", "mu_lib, sigma_lib", "4000, 0.35", FIT, None,
         (f"{_R}:40",), (f"{_ER}:324",)),
    Bias("sequencing", ("Visium",), "spot depth", "mu_counts, sigma_counts", "20000, 0.45", FIT, None,
         (f"{_B}:917", f"{_B}:918"), (f"{_EV}:614",)),
    # --- spatial: Visium only (the dissociated assays lose cell positions) ----------------------------
    Bias("spatial", ("Visium",), "capture field", "field_sigma, field_lengthscale", "0.70, 18", FIT, None,
         (f"{_B}:919", f"{_B}:920"), (f"{_EV}:616", f"{_EV}:619")),
    Bias("spatial", ("Visium",), "edge falloff", "edge_sigma", "0.30", COND, "needs off-tissue spots",
         (f"{_B}:921",), (f"{_EV}:586",)),
    Bias("spatial", ("Visium",), "spot bleed", "bleed_rate", "0 (off)", COND, "needs off-tissue spots",
         (f"{_B}:924",), (f"{_EV}:579",)),
    Bias("spatial", ("Visium",), "lateral diffusion", "diffusion_sigma", "0", SET, None,
         (f"{_B}:923",), (f"{_EV}:64",)),
    # --- batch: what differs between two runs of the same assay ---------------------------------------
    Bias("batch", ("WGS", "scDNA"), "depth shift", "depth_batch_sigma", "0.05", COND, ">=2 batches",
         (f"{_B}:705",), (f"{_ED}:511",)),
    Bias("batch", ("scRNA",), "per-gene factor", "sigma_batch", "0.10", COND, ">=2 batches",
         (f"{_R}:40",), (f"{_ER}:343",)),
    Bias("batch", ("scRNA",), "depth shift", "depth_batch_sigma", "0.05", COND, ">=2 batches",
         (f"{_R}:43",), (f"{_ER}:343",)),
    Bias("batch", ("scRNA",), "plate depth", "plate_sigma, well_sigma", "0, 0.15 (Smart-seq3)", SET, None,
         (f"{_R}:53",), (f"{_ER}:61",)),
    Bias("batch", ("Visium",), "per-gene factor", "sigma_batch", "0.10", COND, ">=2 sections",
         (f"{_B}:927",), (f"{_EV}:641",)),
]


# What the figure prints for each entry of BIASES, keyed by (bias, param, default): the quantity the
# parameter sets and its default, in words, and for a conditional fit what the real data must carry.
PLAIN = {
    ("cell-type recovery", "recovery_probs", "immune 0.4 ... cancer 0.9"):
        "recovery probability, immune 0.4 to cancer 0.9",
    ("dissociation stress", "stress_strength", "0 (random 5% of genes)"): "strength 0, on a random 5% of genes",
    ("section thickness", "depth_frac", "0.5"): "share of the depth kept 0.5",
    ("GC bias", "gc_curve_sigma", "0.20 (panel 0.05)"): "curve strength 0.20, panel 0.05",
    ("capture variability", "capture_sigma", "0 (panel 0.45)"): "log-normal SD 0, panel 0.45",
    ("FFPE C>T damage", "ffpe_ct_rate", "0"): "rate 0",
    ("amplification lumpiness", "kappa", "5"): "concentration 5",
    ("allelic dropout", "ado_rate", "0.20"): "rate 0.20",
    ("allelic imbalance", "beta_binom_conc", "30"): "beta-binomial concentration 30",
    ("doublets", "doublet_rate", "0.02"): "rate 0.02",
    ("count overdispersion", "dispersion", "0.30"): "dispersion 0.30",
    ("dropout", "dropout_mid", "1.5 (Smart-seq3 0)"): "midpoint 1.5, Smart-seq3 0",
    ("ambient RNA", "ambient_frac", "0.05"): "share 0.05",
    ("doublets", "doublet_rate", "0.05"): "rate 0.05",
    ("count overdispersion", "kappa", "50"): "concentration 50",
    ("depth", "mu_depth", "30 (x0.3 single cell)"): "mean 30x, single cells 9x",
    ("depth dispersion", "kappa", "2000"): "concentration 2000",
    ("base errors", "error_rate", "0.001"): "rate 0.001",
    ("library size", "mu_lib, sigma_lib", "4000, 0.35"): "mean 4000, log-normal SD 0.35",
    ("spot depth", "mu_counts, sigma_counts", "20000, 0.45"): "mean 20000, log-normal SD 0.45",
    ("capture field", "field_sigma, field_lengthscale", "0.70, 18"): "strength 0.70, length scale 18",
    ("edge falloff", "edge_sigma", "0.30"): "strength 0.30",
    ("spot bleed", "bleed_rate", "0 (off)"): "rate 0",
    ("lateral diffusion", "diffusion_sigma", "0"): "kernel SD 0",
    ("depth shift", "depth_batch_sigma", "0.05"): "log-normal SD 0.05",
    ("per-gene factor", "sigma_batch", "0.10"): "log-normal SD 0.10",
    ("plate depth", "plate_sigma, well_sigma", "0, 0.15 (Smart-seq3)"): "plate SD 0, well SD 0.15 in Smart-seq3",
}
FITTED_GIVEN = {
    "needs GC": "fitted given GC content", "WES or panel": "fitted on exome or panel data",
    "needs C>T mask": "fitted given C>T sites", "needs het sites": "fitted given heterozygous sites",
    "fit_dropout=True": "fitted on request", "needs variant mask": "fitted given non-variant sites",
    "needs off-tissue spots": "fitted given spots outside the tissue",
    ">=2 batches": "fitted given two batches or more", ">=2 sections": "fitted given two sections or more",
}


def biases(step=None, tech=None):
    """The entries of :data:`BIASES` for one step and/or one technology (Fig 1c's strip uses this)."""
    return [b for b in BIASES if (step is None or b.step == step) and (tech is None or tech in b.techs)]


def check(repo=REPO):
    """Every ``path:line`` of :data:`BIASES` still mentions its parameter; returns the mismatches."""
    bad = []
    for b in BIASES:
        if (b.bias, b.param, b.default) not in PLAIN or (b.how == COND and b.cond not in FITTED_GIVEN):
            bad.append(f"{b.bias} [{', '.join(b.techs)}]: no plain text in PLAIN / FITTED_GIVEN")
        tokens = [p.strip() for p in b.param.split(",")]
        if b.param == "recovery_probs":
            tokens = ["DEFAULT_RECOVERY"]
        for kind, locs in (("src", b.src), ("fit_src", b.fit_src or ())):
            if not locs:
                continue
            text = ""
            for loc in locs:
                path, line = loc.rsplit(":", 1)
                with open(os.path.join(repo, path)) as f:
                    lines = f.read().splitlines()
                text += lines[int(line) - 1] if int(line) <= len(lines) else ""
            missing = [t for t in tokens if t not in text]
            if missing:
                bad.append(f"{b.bias} [{', '.join(b.techs)}] {kind} {locs}: no {missing}")
    return bad


# --- render --------------------------------------------------------------------------------------
def _mark(ax, x, y, how, colour, size=3.2):
    fill = {FIT: "full", COND: "left", SET: "none"}[how]
    ax.plot([x], [y], marker="o", ms=size, fillstyle=fill, color=colour, mec=colour, mew=0.6,
            markerfacecoloralt="white" if fill == "left" else colour, clip_on=False)


def render():
    sys.path.insert(0, HERE)
    import style as S
    S.setup()
    col_tech = {"WGS": S.MODALITY["DNA"], "scDNA": S.MODALITY["DNA"], "scRNA": S.MODALITY["RNA"],
                "Visium": S.MODALITY["spatial"]}
    x0, cw, gap, line_h, wrap = 19.0, 39.0, 1.5, 2.6, 37
    cells = {}
    for st in STEPS:
        for te in TECHS:
            items = []
            for b in biases(st, te):
                txt = f"{b.bias}: {PLAIN[b.bias, b.param, b.default]}"
                if b.how == COND:
                    txt += f", {FITTED_GIVEN[b.cond]}"
                items.append((b.how, textwrap.wrap(txt, wrap, subsequent_indent="  ")))
            cells[st, te] = items
    heights = {st: max(2.0 + line_h * max(1, sum(len(w) for _, w in cells[st, te])) for te in TECHS) for st in STEPS}
    H = 12 + sum(heights[st] + gap for st in STEPS) + 18
    fig = S.figure("double", H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, S.WIDTH_MM["double"])
    ax.set_ylim(H, 0)
    ax.axis("off")
    for j, te in enumerate(TECHS):
        xc = x0 + j * (cw + gap)
        ax.text(xc + 1.5, 8.5, te, fontsize=7, fontweight="bold", color=col_tech[te], va="bottom")
        ax.plot([xc, xc + cw], [9.5, 9.5], color=col_tech[te], lw=1.2, solid_capstyle="butt")
    y = 11.5
    for st in STEPS:
        h = heights[st]
        ax.text(2, y + 2.4, st, fontsize=6.5, fontweight="bold", va="top")
        for j, te in enumerate(TECHS):
            xc = x0 + j * (cw + gap)
            ax.add_patch(__import__("matplotlib").patches.Rectangle((xc, y), cw, h, color=S.PASTEL["grey"], lw=0))
            items = cells[st, te]
            if not items:
                reason = "dissociated: no positions" if st == "spatial" else "none"
                ax.text(xc + 2, y + 2.4, reason, fontsize=4.8, va="top", color=S.GREY)
                continue
            yy = y + 2.4
            for how, lines in items:
                _mark(ax, xc + 2.2, yy + 0.95, how, col_tech[te])
                for k, ln in enumerate(lines):
                    ax.text(xc + 4.0, yy, ln, fontsize=4.8, va="top")
                    yy += line_h
        y += h + gap
    for k, (how, label) in enumerate(((FIT, "fitted from real data"),
                                      (COND, "fitted when the real data contain what the fit needs"),
                                      (SET, "set by the user as a prior or preset"))):
        _mark(ax, x0 + 1.2, y + 3.0 + 3.0 * k, how, S.INK)
        ax.text(x0 + 3.0, y + 3.0 + 3.0 * k, label, fontsize=5, va="center")
    path = S.save(fig, "supp_biases", stamp=False)
    print("wrote", path)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify every path:line and exit")
    a = ap.parse_args()
    bad = check()
    for m in bad:
        print("MISMATCH", m)
    print(f"checked {len(BIASES)} entries: {len(bad)} mismatches")
    if a.check:
        sys.exit(1 if bad else 0)
    render()


if __name__ == "__main__":
    main()
