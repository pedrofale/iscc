#!/usr/bin/env python
"""Grow the tumours and the cohort of ``iscc.pregrown`` and write the package's manifest.

Every tumour is the paper's regime (``regime.py`` in this folder) on the arm-level human genome, as
the registry's run R11 grows it, at the cm scale (grid 170) to 200,000 cancer cells. ``ductal`` is
that tumour; ``immune`` is it with the regime's immune cells, ``neutral``, ``sweeps`` and ``cin`` are variants
of ``immune``, and ``cohort`` is six
patients of ``immune``'s configuration (immune cells are the commonest state shared across patients, which
integration across patients should keep) at the mid scale (grid 96) to 60,000 cancer cells each. Each object is saved with its per-cell tables built, as
``grow`` leaves a tumour, so that a loaded tumour is in the same state.

Usage:
    python validation/paper/build_pregrown.py grow <out_dir> [name ...] [--scale mid --target 60000]
                                     [--cohort-scale small --cohort-target 8000]
    python validation/paper/build_pregrown.py manifest <out_dir> [--base-url URL]

``grow`` writes ``<name>.iscc``, a summary ``<name>.json`` and a figure ``<name>.png`` (the tissue
and the Muller plot) per name; ``--scale`` and ``--target`` grow smaller trial versions. ``manifest``
writes the checksums of the files in ``<out_dir>`` into ``src/iscc/pregrown/manifest.json``.
"""
import argparse
import functools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
os.environ.setdefault("ISCC_PAPER_TIER", "full")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, HERE)
import regime as RG  # noqa: E402
import registry as REG  # noqa: E402
from iscc import pregrown  # noqa: E402
from iscc.cohort import Cohort  # noqa: E402
from iscc.data.human import arm_genome  # noqa: E402
from iscc.data.human.reference import _sha256  # noqa: E402

SCALE, TARGET = "cm", 200_000
# Cohort keeps every patient in memory, and the regime's cm-scale tumour peaks at 7.4 GB (R11 on
# Arrhenius), so the cohort grows at the mid scale (grid 96), the scale of the paper's seeding runs.
COHORT_SCALE, COHORT_TARGET = "mid", 60_000
MANIFEST = os.path.join(REPO, "src", "iscc", "pregrown", "manifest.json")


def _arm(driver_scale=0.4):
    """R11's arm-genome blocks: the genome, the drivers placed per arm, and the copy-number rates
    scaled to 39 arms (registry.py documents the scaling)."""
    ag = arm_genome(total_genes=6000, driver_scale=driver_scale)
    rates = REG.RUNS["R11"][0][1]["cancer"]
    return dict(genome=ag["genome_params"], selection=dict(ag["selection_params"]), cancer=dict(rates))


# The density of germline sites of a genotyping array (about 30% heterozygous), so that the DNA
# assays carry the B-allele signal allele-specific callers read; the regime's 0.5% leaves about 30
# sites on the growth genome. The germline changes no growth statistic (Cell._available_sites).
GERMLINE = {"het_frac": 0.3, "hom_frac": 0.33}


def _variant(selection=None, cancer=None, spatial=None, immune=None, driver_scale=0.4):
    kw = _arm(driver_scale)
    kw["germline"] = dict(GERMLINE)
    kw["selection"].update(selection or {})
    kw["cancer"].update(cancer or {})
    if spatial:
        kw["spatial"] = dict(spatial)
    if immune:
        kw["immune"] = dict(immune)
    return kw


# name -> (seed, regime overrides). Seed 1 of R11 is swept by a genome doubling, so ductal is seed 2.
# ductal is the paper's tumour, without immune cells. Every other tumour has the regime's immune
# variant (regime.IMMUNE_SPATIAL, IMMUNE_SELECTION, IMMUNE, read out with the interferon response), so
# that the tutorials, which load these, show the most complete simulation (Pedro, 2026-11-22): immune is
# ductal with immune cells, and sweeps and cin are its variants. neutral has no selection at all:
# drivers without effect, no niche traits, the duct wall and the stroma open to every cell (R10's "open"
# arm), and immune cells that kill nothing, since immune killing selects resistance. With the barriers
# on, a tumour without drivers never selects breach against its cost and stays in its glands (a smoke
# run held 6,359 cells after 1,500 generations).
IMMUNE = dict(selection=RG.IMMUNE_SELECTION, spatial=RG.IMMUNE_SPATIAL, immune=RG.IMMUNE)
TUMOURS = {
    "ductal": (2, _variant()),
    "immune": (2, _variant(**IMMUNE)),
    "neutral": (2, _variant(selection={"driver_effects": 1.0, "prop_breach": 0.0,
                                       "prop_stromal_survival": 0.0, "prop_immune_resistance": 0.0},
                            spatial={**RG.IMMUNE_SPATIAL, "breach_gated_invasion": False,
                                     "stromal_hazard": 0.0},
                            immune={"prob_kill": 0.0})),
    "sweeps": (2, _variant(selection={**RG.IMMUNE_SELECTION, "driver_effects": 1.15},
                           spatial=RG.IMMUNE_SPATIAL, immune=RG.IMMUNE, driver_scale=0.1)),
    "cin": (2, _variant(cancer={"cnv_prob": 4 * _arm()["cancer"]["cnv_prob"],
                                "wgd_rate": 5 * _arm()["cancer"]["wgd_rate"]}, **IMMUNE)),
}
COHORT_SEEDS = [11, 12, 13, 14, 15, 16]


def _grow(seed, kw, scale, target, layout_seed=None):
    t = RG.grow(seed=seed, scale=scale, target_cancer=target, spatial_snapshot_every=8,
                layout_seed=layout_seed, **kw)
    t.make_cell_data()
    return t


def _cancer_shares(t):
    """Over the cancer cells: the share with a doubled genome, the share with a segment at copy number
    0, and the mean ploidy (segments weighted by their genes)."""
    import numpy as np
    sizes = np.asarray(t.selection.segment_sizes, dtype=float)
    n = wgd = null = ploidy = 0.0
    for g, c in t.genotypes_counts.items():
        if c <= 0 or not t._is_cancer(g):
            continue
        rep = t.genotypes[g]
        cns = np.asarray(rep.genome_summary["seg_cns"], dtype=float)
        n += c
        wgd += c * bool(rep.is_wgd)
        null += c * bool((cns == 0).any())
        ploidy += c * float((cns * sizes).sum() / sizes.sum())
    return {"wgd_share": round(wgd / n, 3), "nullisomic_share": round(null / n, 3),
            "mean_ploidy": round(ploidy / n, 2)}


def _summary(t, name, seconds):
    d = t.diagnose(verbose=False)
    return {"name": name, "cancer_cells": int(t.get_cancer_size()), "generations": round(t.time, 1),
            "reached_target": bool(t.reached_target), "genotypes": len(t.genotypes),
            "grow_s": round(seconds), **_cancer_shares(t),
            "flags": [f for f, v in getattr(d, "flags", {}).items() if v]}


def _figure(t, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.6]})
    t.plot_tissue(color="clone", ax=axes[0])
    t.plot_muller(ax=axes[1], by_drivers=True, min_freq=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _check_loads(path, kind):
    """Load the file in a fresh process that sees the package only, as a user's would."""
    code = (f"from iscc.{'cohort' if kind == 'Cohort' else 'tumor'} import {kind}; "
            f"print(type({kind}.load({path!r})).__name__)")
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    out = subprocess.run([sys.executable, "-s", "-c", code], capture_output=True, text=True, env=env,
                         cwd="/")
    if out.returncode or out.stdout.strip() != kind:
        raise RuntimeError(f"{path} does not load in a fresh process:\n{out.stderr[-2000:]}")


def grow(out_dir, names, scale, target, cohort_scale=COHORT_SCALE, cohort_target=COHORT_TARGET):
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        t0 = time.time()
        if name == "cohort":
            seed, kw = TUMOURS["immune"]
            at_scale, at_target = cohort_scale, cohort_target
            obj = Cohort(functools.partial(_grow, kw=kw, scale=at_scale, target=at_target),
                         patient_seeds=COHORT_SEEDS).run()
            # The growth function lives in validation/, which an installed iscc does not have.
            obj.grow = pregrown._grown_elsewhere
            summary = {"name": name, "patients": [_summary(p.tumor, f"patient {p.patient_id}", 0)
                                                  for p in obj.patients]}
        else:
            seed, kw = TUMOURS[name]
            at_scale, at_target = scale, target
            obj = _grow(seed, kw, scale, target)
            summary = _summary(obj, name, time.time() - t0)
            _figure(obj, os.path.join(out_dir, f"{name}.png"))
        path = os.path.join(out_dir, f"{name}.iscc")
        obj.save(path)
        kind = type(obj).__name__
        _check_loads(path, kind)
        summary.update(scale=at_scale, target=at_target, kind=kind, bytes=os.path.getsize(path),
                       total_s=round(time.time() - t0))
        with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(json.dumps(summary), flush=True)


def manifest(out_dir, base_url):
    files = {}
    for name in [*TUMOURS, "cohort"]:
        path = os.path.join(out_dir, f"{name}.iscc")
        if not os.path.exists(path):
            continue
        with open(os.path.join(out_dir, f"{name}.json")) as f:
            kind = json.load(f)["kind"]
        files[name] = {"file": f"{name}.iscc", "sha256": _sha256(path), "bytes": os.path.getsize(path),
                       "kind": kind}
    with open(MANIFEST, "w") as f:
        json.dump({"base_url": base_url, "files": files}, f, indent=2)
        f.write("\n")
    print(f"{len(files)} files in {MANIFEST}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["grow", "manifest"])
    ap.add_argument("out_dir")
    ap.add_argument("names", nargs="*")
    ap.add_argument("--scale", default=SCALE)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--cohort-scale", default=COHORT_SCALE)
    ap.add_argument("--cohort-target", type=int, default=COHORT_TARGET)
    ap.add_argument("--base-url", default=None)
    a = ap.parse_args()
    if a.action == "grow":
        grow(a.out_dir, a.names or [*TUMOURS, "cohort"], a.scale, a.target, a.cohort_scale,
             a.cohort_target)
    else:
        manifest(a.out_dir, a.base_url)
