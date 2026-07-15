"""Benchmark: can a program-inference tool recover iscc's TRUE gene programs, and how does that
degrade with SNV/CNA burden? (R13, `DESIGN_expression.md` §4.2)

iscc knows the true `loading` matrix and the true per-cell activity `z` (`tumor.program_truth`), so
it can ask a question no real dataset can: not "do these factors look biological?" but "are they the
right ones?" — scored against the answer key, across a genotype-burden sweep.

**Flagship: scDEF** (hierarchical Bayesian factor model). **Comparator: cNMF** (the field-standard
consensus-NMF GEP method). Each in its own dedicated env (`iscc-scdef`, `iscc-cnmf`), per
`README_integration.md`; this script stays in the core env and shells out.

THE HYPOTHESIS UNDER TEST — stated up front so a null result is still a result:

    CNAs are CONTIGUOUS. A high CNA burden should therefore induce *positional* co-expression —
    genes co-varying because they share a copy-number segment, not because they share a function. A
    factor model has no notion of genome position, so it can absorb that as spurious "programs", and
    true-program recovery should degrade as fraction-genome-altered rises. The DISCRIMINATING
    diagnostic is the orthogonality itself (programs |= CNAs): are a factor's top genes positionally
    CLUSTERED (a CNA artefact) or SCATTERED (a real program)?

    If it reproduces, it is a direct sibling of the PEtracer lineage-space confound — *genotype
    structure confounds expression-program inference* — and belongs in the same "structure misleads
    inference" arc as PEtracer and multi-region. If CNA burden does NOT degrade recovery, that is
    itself a reportable (and reassuring) result about the tools, and is reported as such. The sweep
    is deliberately NOT tuned to manufacture the effect.

Why this is non-circular: iscc's forward model is NOT the law these tools invert. Programs are
FUNCTIONAL gene sets scattered genome-wide; dosage is per-gene BUFFERED (`s_g`), not linear; the SNV
expression effect is decoupled from fitness; and route 1 drives programs from the evolved phenotype,
so clone identity leaks into expression through a NON-dosage channel. All four are confounders the
tools' own assumptions do not contain.

Panels (figure manuscript/figures/validation_programs.png):
  A. The ANSWER KEY: the true loading matrix — scattered (default) vs the CNA-mimicking control
     (`program_genomic_scatter=0`).
  B. Recovery vs CNA burden (fraction-genome-altered): loading cosine + activity correlation.
  C. Recovery vs SNV burden.
  D. THE DIAGNOSTIC: positional clustering of matched vs spurious factors, against FGA.

Usage:
    python validate_programs.py                 # full sweep (slow: fits both tools per condition)
    python validate_programs.py --quick         # 1 rep, coarse grid — smoke test
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import programs_common as pc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL_COLORS = {"scdef": "#4C72B0", "cnmf": "#DD8452"}


def one_condition(tool, seed, steps, mutation_rate=None, cnv_prob=None, amp_prob=None,
                  scatter=1.0, n_epoch=300, n_iter=20, k_factors=None):
    """Grow one tumour, run one tool, score it. Returns None if the tool's env is absent.

    `k_factors` defaults to 2x the true program count ON PURPOSE: a tool given exactly K_true factors
    has no room to invent a spurious one, so "did CNA burden create spurious programs?" would be
    unanswerable by construction. The headroom is where CNA absorption becomes visible.
    """
    t = pc.grow_tumor(seed=seed, steps=steps, mutation_rate=mutation_rate, cnv_prob=cnv_prob,
                      amp_prob=amp_prob, scatter=scatter)
    if t.get_cancer_size() < 100:
        return None                      # extinct/tiny: nothing to fit, don't fabricate a score
    adata, Z = pc.counts_anndata(t, seed=seed)
    k = k_factors or 2 * pc.N_PROGRAMS
    inferred = pc.run_tool(tool, adata, k=k, seed=seed, n_epoch=n_epoch, n_iter=n_iter)
    if inferred is None:
        return None
    s = pc.score_recovery(t.program_truth["loading"], Z, inferred, t.selection.segment_sizes)
    s.update(fga=pc.fga(t), snv=pc.snv_burden(t), n_cancer=int(t.get_cancer_size()))
    return s


def sweep(tools, values, key, reps, steps, n_epoch, n_iter, k_factors=None, **fixed):
    """Sweep one burden knob, averaging over replicate seeds. Returns {tool: [row-per-value]}."""
    out = {tool: [] for tool in tools}
    for v in values:
        for tool in tools:
            runs = []
            for r in range(reps):
                kw = dict(fixed); kw[key] = v
                try:
                    s = one_condition(tool, seed=1 + r, steps=steps, n_epoch=n_epoch,
                                      n_iter=n_iter, k_factors=k_factors, **kw)
                except Exception as e:
                    print(f"  [{tool}] {key}={v} rep{r} FAILED: {str(e)[:200]}")
                    continue
                if s is not None:
                    runs.append(s)
            if not runs:
                continue
            agg = {m: float(np.nanmean([r_[m] for r_ in runs]))
                   for m in ("loading_cosine", "gene_jaccard", "activity_corr", "n_spurious",
                             "clustering_matched", "clustering_spurious", "fga", "snv")}
            agg[key] = v
            agg["n_reps"] = len(runs)
            out[tool].append(agg)
            print(f"  [{tool}] {key}={v}: FGA={agg['fga']:.3f} SNV={agg['snv']:.3f} "
                  f"cos={agg['loading_cosine']:.3f} act={agg['activity_corr']:.3f} "
                  f"spurious={agg['n_spurious']:.1f} clust_m={agg['clustering_matched']:.2f} "
                  f"clust_s={agg['clustering_spurious']:.2f} (n={len(runs)})")
    return out


def _plot_sweep(ax, res, xkey, ykey, ylabel, title):
    for tool, rows in res.items():
        if not rows:
            continue
        x = [r[xkey] for r in rows]
        y = [r[ykey] for r in rows]
        ax.plot(x, y, "o-", color=TOOL_COLORS.get(tool, "grey"), label=tool)
    ax.set_xlabel(xkey.upper() if xkey == "fga" else "SNV burden (frac. sites mutated)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def make_figure(truth_scattered, truth_clustered, cna_res, snv_res, control_res, out):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    seg_sizes = [pc.GENOME["segment_size"]] * pc.GENOME["n_segments"]
    null_mean, null_p95 = pc.scattered_null(seg_sizes)

    # A1/A2 — the answer key, and the CNA-mimicking control
    for ax, M, ttl in ((axes[0, 0], truth_scattered,
                        "A1. Truth: scattered programs (default) — programs ⟂ CNAs"),
                       (axes[1, 0], truth_clustered,
                        "A2. Control: program_genomic_scatter=0 — a program that MIMICS a CNA")):
        ax.imshow(np.abs(M) > 0, aspect="auto", cmap="Greys", interpolation="nearest")
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("gene (genome order)"); ax.set_ylabel("program")
        for off in np.cumsum(seg_sizes)[:-1]:
            ax.axvline(off, color="tab:red", lw=0.4, alpha=0.5)

    # B — recovery vs CNA burden
    _plot_sweep(axes[0, 1], cna_res, "fga", "loading_cosine", "loading cosine vs truth",
                "B. True-program recovery vs CNA burden")
    for tool, rows in cna_res.items():
        if rows:
            axes[0, 1].plot([r["fga"] for r in rows], [r["activity_corr"] for r in rows], "s--",
                            color=TOOL_COLORS.get(tool, "grey"), alpha=0.5,
                            label=f"{tool} (activity)")
    axes[0, 1].legend(fontsize=7)

    # C — recovery vs SNV burden (the control axis: SNVs are scattered, so nothing should happen)
    _plot_sweep(axes[1, 1], snv_res, "snv", "loading_cosine", "loading cosine vs truth",
                "C. True-program recovery vs SNV burden (control axis)")

    # D — THE diagnostic
    ax = axes[0, 2]
    for tool, rows in cna_res.items():
        if not rows:
            continue
        x = [r["fga"] for r in rows]
        ax.plot(x, [r["clustering_matched"] for r in rows], "o-",
                color=TOOL_COLORS.get(tool, "grey"), label=f"{tool}: real factors")
        ax.plot(x, [r["clustering_spurious"] for r in rows], "x--",
                color=TOOL_COLORS.get(tool, "grey"), alpha=0.7, label=f"{tool}: SPURIOUS factors")
    ax.axhline(null_mean, color="k", ls=":", label=f"scattered null ({null_mean:.2f})")
    ax.axhline(null_p95, color="k", ls="--", alpha=0.4, label="scattered null 95th pct")
    ax.set_xlabel("FGA"); ax.set_ylabel("max frac. of top genes in ONE segment")
    ax.set_title("D. Diagnostic: positionally clustered (CNA) or scattered (real)?", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

    # E — how many spurious factors, and the scatter=0 control's honest limitation
    ax = axes[1, 2]
    for tool, rows in cna_res.items():
        if rows:
            ax.plot([r["fga"] for r in rows], [r["n_spurious"] for r in rows], "o-",
                    color=TOOL_COLORS.get(tool, "grey"), label=f"{tool}: # spurious factors")
    for tool, rows in (control_res or {}).items():
        for r in rows:
            ax.axhline(r["n_spurious"], color=TOOL_COLORS.get(tool, "grey"), ls=":", alpha=0.6,
                       label=f"{tool}: scatter=0 control")
    ax.set_xlabel("FGA"); ax.set_ylabel("# factors matching NO true program")
    ax.set_title("E. Spurious factors invented, vs CNA burden", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.suptitle("Gene-program recovery vs genotype burden (scDEF vs cNMF) — iscc R13", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "manuscript/figures/validation_programs.png"))
    ap.add_argument("--cnv-probs", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--mutation-rates", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--n-epoch", type=int, default=300)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--k-factors", type=int, default=2 * pc.N_PROGRAMS,
                    help="factors given to each tool; > n_programs so spurious factors CAN appear")
    ap.add_argument("--tools", nargs="+", default=["scdef", "cnmf"])
    ap.add_argument("--quick", action="store_true", help="1 rep, coarse grid, fewer epochs")
    args = ap.parse_args()

    if args.quick:
        args.reps, args.cnv_probs, args.mutation_rates = 1, [0.0, 0.75], [0.1, 2.0]
        args.n_epoch, args.n_iter, args.steps = 100, 8, 8000

    tools = [t for t in args.tools
             if (t == "scdef" and pc.scdef_available()) or (t == "cnmf" and pc.cnmf_available())]
    missing = set(args.tools) - set(tools)
    if missing:
        print(f"SKIPPING (env absent): {sorted(missing)} — see validation/README_integration.md")
    if not tools:
        print("no tool envs available; nothing to do")
        return

    # The ground-truth panel + the CNA-mimicking control (program_genomic_scatter=0).
    t_scat = pc.grow_tumor(seed=1, steps=200, scatter=1.0)
    t_clust = pc.grow_tumor(seed=1, steps=200, scatter=0.0)

    print("\n=== CNA burden sweep (cnv_prob -> FGA) ===")
    cna_res = sweep(tools, args.cnv_probs, "cnv_prob", args.reps, args.steps,
                    args.n_epoch, args.n_iter, k_factors=args.k_factors)
    print("\n=== SNV burden sweep (mutation_rate) ===")
    snv_res = sweep(tools, args.mutation_rates, "mutation_rate", args.reps, args.steps,
                    args.n_epoch, args.n_iter, k_factors=args.k_factors, cnv_prob=0.0)

    # CONTROL: a deliberately CNA-mimicking PROGRAM (program_genomic_scatter=0) at zero CNA burden.
    # The honest question the positional diagnostic must face: it calls a factor an artefact when its
    # genes are contiguous — so what does it do with a program that is genuinely contiguous? If the
    # diagnostic flags this one too, it is detecting POSITION, not artefact, and that is its stated
    # limitation rather than a bug.
    print("\n=== CONTROL: CNA-mimicking program (program_genomic_scatter=0, no CNAs) ===")
    control_res = sweep(tools, [0.0], "scatter", args.reps, args.steps,
                        args.n_epoch, args.n_iter, k_factors=args.k_factors, cnv_prob=0.0)

    make_figure(t_scat.program_truth["loading"], t_clust.program_truth["loading"],
                cna_res, snv_res, control_res, args.out)


if __name__ == "__main__":
    main()
