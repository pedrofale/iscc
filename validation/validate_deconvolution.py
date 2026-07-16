"""Spatial-deconvolution benchmark (cell2location + RCTD) — the FLAGSHIP multi-modal integration demo.

`iscc` emits BOTH a Visium section AND an scRNA reference FROM THE SAME TUMOUR, and knows the TRUE
per-spot cell-type composition of the section *and* the true composition of the reference sample.
Real spatial-deconvolution benchmarks must borrow a reference from a different piece of tissue and can
never measure the resulting reference mismatch, let alone separate its causes. We run the GENUINE tools
— cell2location (`iscc-cell2location`) and RCTD/spacexr (`iscc-rctd`), each in its own env — and score
per-spot proportions against `iscc`'s ground truth.

WHY THIS IS NON-CIRCULAR. The four cell types differ transcriptionally because the R13 program layer
gives each a distinct COMBINATION of functional programmes (scattered gene sets), an emergent property
of the tumour — not a hand-drawn marker table matching what the deconvolution model assumes. And the
copy-number clones differ only by contiguous dosage, so a CNA clone is NOT a program combination —
which is exactly why the tools should (and do) confuse clones with cell types far less well than they
resolve the cell types themselves.

THE HEADLINE (panel A). What does a realistic (same-tumour, different-sample) reference cost, split
into its three real, separable causes? An ORACLE reference from the exact section cells is the ceiling;
we then add (1) REGIONAL mismatch (reference from a different region), (2) DISSOCIATION bias (F2),
(3) ASSAY/batch mismatch (F3), and attribute the error to each. This makes deconvolution a downstream
consequence of the paper's "Biopsy and dissociation shape the sampled data" section — the same sampling
biases, now with ground truth at both ends.

Figure (manuscript/figures/validation_deconvolution.png):
  A. reference-mismatch decomposition (oracle -> +regional -> +dissociation -> +assay), both tools;
  B. accuracy vs cells-per-spot (spot size sweep);
  C. regional-mismatch sweep — accuracy AND the measured reference-composition error vs biopsy offset;
  D. clone-vs-cell-type confound — the tools resolve program-defined cell types but not CNA clones.

Run (from the repo root, in the `iscc` env):
    python -u validation/validate_deconvolution.py            # the paper figure
    python -u validation/validate_deconvolution.py --quick    # fast smoke (RCTD only, few points)
Requires `iscc-cell2location` and/or `iscc-rctd`; missing tools are skipped with a printed note.
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deconv_common as D

REPO = D.REPO
FIG = os.path.join(REPO, "manuscript", "figures", "validation_deconvolution.png")


def _run_tool(tool, ref, section, work_dir, target="type", **kw):
    """Run one tool and score against the section truth for the chosen target (type|clone)."""
    if tool == "rctd":
        props, cats = D.run_rctd(ref, section, work_dir, **kw)
    else:
        props, cats = D.run_cell2location(ref, section, work_dir, **kw)
    true = section["true_type"] if target == "type" else section["true_clone"]
    catset = section["type_categories"] if target == "type" else section["clone_categories"]
    return D.score_proportions(true, props.values, catset, cats)


def experiment_decomposition(tumor, section, tools, tmp, epochs, epochs_sp, seed=0):
    """Panel A: oracle -> +regional -> +dissociation -> +assay, for each tool."""
    scenarios = [
        ("oracle", dict(mode="oracle", dissociation_strength=0.0, protocol="10x")),
        ("+regional", dict(mode="regional", offset=10.0, dissociation_strength=0.0, protocol="10x")),
        ("+dissoc", dict(mode="regional", offset=10.0, dissociation_strength=1.0, protocol="10x")),
        ("+assay", dict(mode="regional", offset=10.0, dissociation_strength=1.0, protocol="smartseq3")),
    ]
    rows = []
    for name, kw in scenarios:
        ref = D.build_reference(tumor, section, n_cells=450, seed=11, **kw)
        for tool in tools:
            wd = os.path.join(tmp, f"decomp_{name}_{tool}")
            tkw = dict(epochs=epochs, epochs_sp=epochs_sp) if tool == "cell2location" else {}
            sc = _run_tool(tool, ref, section, wd, target="type", **tkw)
            rows.append(dict(scenario=name, tool=tool, jsd=sc["jsd"], rmse=sc["rmse"],
                             r=sc["flat_r"], cancer_frac=ref["composition"].get("cancer", 0.0)))
            print(f"  [A] {tool:14s} {name:10s} JSD={sc['jsd']:.3f} r={sc['flat_r']:.3f} "
                  f"ref_cancer={ref['composition'].get('cancer', 0):.2f}")
    return pd.DataFrame(rows)


def experiment_cells_per_spot(tumor, tools, tmp, epochs, epochs_sp, radii, seed=0):
    """Panel B: accuracy vs cells-per-spot (sweep the Visium spot radius). Oracle reference isolates
    the spot-mixing difficulty from reference mismatch."""
    rows = []
    for rad in radii:
        section = D.build_section(tumor, spot_radius=rad, spot_pitch=1.5, section_radius=9.0)
        cps = float(np.mean(section["n_cells"]))
        ref = D.build_reference(tumor, section, mode="oracle", n_cells=450, seed=11)
        for tool in tools:
            wd = os.path.join(tmp, f"cps_{rad}_{tool}")
            tkw = dict(epochs=epochs, epochs_sp=epochs_sp) if tool == "cell2location" else {}
            sc = _run_tool(tool, ref, section, wd, target="type", **tkw)
            rows.append(dict(radius=rad, cells_per_spot=cps, tool=tool, jsd=sc["jsd"],
                             r=sc["flat_r"], n_spots=sc["n_spots"]))
            print(f"  [B] {tool:14s} radius={rad:.2f} cps={cps:4.1f} JSD={sc['jsd']:.3f} r={sc['flat_r']:.3f}")
    return pd.DataFrame(rows)


def experiment_regional_sweep(tumor, section, tools, tmp, epochs, epochs_sp, offsets, seed=0):
    """Panel C: sweep the reference biopsy offset — accuracy AND the measured reference-composition
    error (L1 vs the section composition), showing the mismatch itself is measurable, not just its cost."""
    sec_comp = np.asarray([section["true_type"].mean(0)[i]
                           for i in range(len(section["type_categories"]))])
    rows = []
    for off in offsets:
        ref = D.build_reference(tumor, section, mode="regional", offset=off,
                                dissociation_strength=0.0, protocol="10x", n_cells=450, seed=11)
        ref_comp = np.asarray([ref["composition"][c] for c in section["type_categories"]])
        comp_l1 = float(np.abs(ref_comp - sec_comp).sum())
        for tool in tools:
            wd = os.path.join(tmp, f"reg_{off}_{tool}")
            tkw = dict(epochs=epochs, epochs_sp=epochs_sp) if tool == "cell2location" else {}
            sc = _run_tool(tool, ref, section, wd, target="type", **tkw)
            rows.append(dict(offset=off, comp_l1=comp_l1, tool=tool, jsd=sc["jsd"], r=sc["flat_r"],
                             ref_cancer=ref["composition"].get("cancer", 0.0)))
            print(f"  [C] {tool:14s} offset={off:4.1f} comp_L1={comp_l1:.2f} "
                  f"ref_cancer={ref['composition'].get('cancer',0):.2f} JSD={sc['jsd']:.3f}")
    return pd.DataFrame(rows)


def experiment_confound(tumor, section, tools, tmp, epochs, epochs_sp, seed=0):
    """Panel D: deconvolve into POPULATIONS (normal types + CNA clones) with an oracle reference. Only
    the cell types are program combinations, so the tools recover them but confuse the CNA clones."""
    ref = D.build_reference(tumor, section, mode="oracle", label_by="population", n_cells=500, seed=11)
    rows = []
    for tool in tools:
        wd = os.path.join(tmp, f"confound_{tool}")
        tkw = dict(epochs=epochs, epochs_sp=epochs_sp) if tool == "cell2location" else {}
        sc = _run_tool(tool, ref, section, wd, target="clone", **tkw)
        for pop, r in sc["per_type_r"].items():
            kind = "CNA clone" if pop.startswith("clone") else "cell type"
            rows.append(dict(tool=tool, population=pop, kind=kind, r=r))
        print(f"  [D] {tool:14s} per-population r={{{', '.join(f'{k}:{v:.2f}' for k,v in sc['per_type_r'].items())}}}")
    return pd.DataFrame(rows)


def make_figure(decomp, cps, regional, confound, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tool_color = {"rctd": "#d1495b", "cell2location": "#2e6f95"}
    tool_label = {"rctd": "RCTD", "cell2location": "cell2location"}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))

    # A. decomposition
    ax = axes[0]
    order = ["oracle", "+regional", "+dissoc", "+assay"]
    if decomp is not None and len(decomp):
        for tool, g in decomp.groupby("tool"):
            g = g.set_index("scenario").reindex(order)
            ax.plot(order, g["jsd"], "-o", color=tool_color[tool], label=tool_label[tool], lw=2)
        ax.set_ylabel("per-spot JSD (lower = better)")
        ax.set_title("A. What an imperfect reference costs\n(decomposed into its three causes)")
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(bottom=0)

    # B. cells per spot
    ax = axes[1]
    if cps is not None and len(cps):
        for tool, g in cps.groupby("tool"):
            g = g.sort_values("cells_per_spot")
            ax.plot(g["cells_per_spot"], g["r"], "-o", color=tool_color[tool], label=tool_label[tool], lw=2)
        ax.set_xlabel("cells per spot")
        ax.set_ylabel("proportion correlation r")
        ax.set_title("B. Accuracy vs spot cellularity\n(oracle reference)")
        ax.legend(frameon=False, fontsize=9)

    # C. regional sweep (accuracy + measured composition mismatch on a twin axis)
    ax = axes[2]
    if regional is not None and len(regional):
        for tool, g in regional.groupby("tool"):
            g = g.sort_values("offset")
            ax.plot(g["offset"], g["jsd"], "-o", color=tool_color[tool], label=tool_label[tool], lw=2)
        ax.set_xlabel("reference biopsy offset (grid units)")
        ax.set_ylabel("per-spot JSD")
        ax.set_title("C. Regional mismatch: cost and cause\n(dashed = measured ref-composition error)")
        ax2 = ax.twinx()
        gg = regional.drop_duplicates("offset").sort_values("offset")
        ax2.plot(gg["offset"], gg["comp_l1"], "--", color="#666666", lw=1.6)
        ax2.set_ylabel("reference composition L1 error", color="#666666")
        ax2.tick_params(axis="y", labelcolor="#666666")
        ax.legend(frameon=False, fontsize=9, loc="upper left")

    # D. clone-vs-celltype confound
    ax = axes[3]
    if confound is not None and len(confound):
        agg = confound.groupby(["tool", "kind"])["r"].mean().reset_index()
        tools = list(agg["tool"].unique())
        kinds = ["cell type", "CNA clone"]
        x = np.arange(len(tools))
        w = 0.35
        for i, kind in enumerate(kinds):
            vals = [agg[(agg.tool == t) & (agg.kind == kind)]["r"].mean() for t in tools]
            ax.bar(x + (i - 0.5) * w, vals, w, label=kind,
                   color="#66a182" if kind == "cell type" else "#c9a227")
        ax.set_xticks(x)
        ax.set_xticklabels([tool_label[t] for t in tools])
        ax.set_ylabel("mean per-population r")
        ax.set_title("D. Cell types are program combinations,\nCNA clones are not")
        ax.legend(frameon=False, fontsize=9)
        ax.axhline(0, color="k", lw=0.6)

    fig.suptitle("iscc as non-circular ground truth for spatial deconvolution: matched vs mismatched reference",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--epochs-sp", type=int, default=3000)
    ap.add_argument("--quick", action="store_true", help="RCTD only, few points, tiny epochs")
    args = ap.parse_args()

    tools = []
    if D.rctd_available():
        tools.append("rctd")
    if D.cell2location_available() and not args.quick:
        tools.append("cell2location")
    if not tools:
        print("No deconvolution tool env found (iscc-rctd / iscc-cell2location) — nothing to run.")
        print("Build them per validation/README_integration.md, then re-run.")
        return

    epochs = 60 if args.quick else args.epochs
    epochs_sp = 300 if args.quick else args.epochs_sp
    radii = [0.7, 1.1] if args.quick else [0.6, 0.8, 1.0, 1.2, 1.5]
    offsets = [0.0, 8.0] if args.quick else [0.0, 4.0, 7.0, 10.0, 13.0]

    print(f"tools: {tools}")
    tumor = D.grow_tumor(seed=args.seed, steps=args.steps)
    types = D.coarse_types(tumor)
    print(f"tumour: n={len(types)} composition={types.value_counts().to_dict()}")
    section = D.build_section(tumor, spot_radius=0.9, spot_pitch=1.5, section_radius=9.0)
    print(f"section: {len(section['spot_names'])} spots, cells/spot={np.mean(section['n_cells']):.1f}, "
          f"mean true comp={dict(zip(section['type_categories'], section['true_type'].mean(0).round(3)))}")

    tmp = tempfile.mkdtemp(prefix="deconv_val_")
    print("\n== A. reference-mismatch decomposition ==")
    decomp = experiment_decomposition(tumor, section, tools, tmp, epochs, epochs_sp, args.seed)
    print("\n== B. accuracy vs cells-per-spot ==")
    cps = experiment_cells_per_spot(tumor, tools, tmp, epochs, epochs_sp, radii, args.seed)
    print("\n== C. regional-mismatch sweep ==")
    regional = experiment_regional_sweep(tumor, section, tools, tmp, epochs, epochs_sp, offsets, args.seed)
    print("\n== D. clone-vs-cell-type confound ==")
    confound = experiment_confound(tumor, section, tools, tmp, epochs, epochs_sp, args.seed)

    make_figure(decomp, cps, regional, confound, FIG)


if __name__ == "__main__":
    main()
