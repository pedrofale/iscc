#!/usr/bin/env python
"""Grow the tumours and the cohort of ``iscc.pregrown`` and write the package's manifest.

Every tumour is the paper's regime (``validation/paper/regime.py``) on the arm-level human genome, as
the registry's run R11 grows it, at the cm scale (grid 170) to 200,000 cancer cells. ``ductal`` is
that tumour; ``neutral``, ``sweeps`` and ``cin`` are declared variants of it, and ``cohort`` is six
patients of ``ductal``'s configuration. Each object is saved with its per-cell tables built, as
``grow`` leaves a tumour, so that a loaded tumour is in the same state.

Usage:
    python scripts/build_pregrown.py grow <out_dir> [name ...] [--scale mid --target 60000]
    python scripts/build_pregrown.py manifest <out_dir> [--base-url URL]

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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ISCC_PAPER_TIER", "full")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "validation", "paper"))
import regime as RG  # noqa: E402
import registry as REG  # noqa: E402
from iscc import pregrown  # noqa: E402
from iscc.cohort import Cohort  # noqa: E402
from iscc.data.human import arm_genome  # noqa: E402
from iscc.data.human.reference import _sha256  # noqa: E402

SCALE, TARGET = "cm", 200_000
MANIFEST = os.path.join(REPO, "src", "iscc", "pregrown", "manifest.json")


def _arm(driver_scale=0.4):
    """R11's arm-genome blocks: the genome, the drivers placed per arm, and the copy-number rates
    scaled to 39 arms (registry.py documents the scaling)."""
    ag = arm_genome(total_genes=6000, driver_scale=driver_scale)
    rates = REG.RUNS["R11"][0][1]["cancer"]
    return dict(genome=ag["genome_params"], selection=dict(ag["selection_params"]), cancer=dict(rates))


def _variant(selection=None, cancer=None, spatial=None, driver_scale=0.4):
    kw = _arm(driver_scale)
    kw["selection"].update(selection or {})
    kw["cancer"].update(cancer or {})
    if spatial:
        kw["spatial"] = dict(spatial)
    return kw


# name -> (seed, regime overrides). Seed 1 of R11 is swept by a genome doubling, so ductal is seed 2.
# neutral has no selection at all: drivers without effect, no niche traits, and the duct wall and the
# stroma open to every cell (R10's "open" arm). With the barriers on, a tumour without drivers never
# selects breach against its cost and stays in its glands (a smoke run held 6,359 cells after 1,500
# generations).
TUMOURS = {
    "ductal": (2, _variant()),
    "neutral": (2, _variant(selection={"driver_effects": 1.0, "prop_breach": 0.0,
                                       "prop_stromal_survival": 0.0, "prop_immune_resistance": 0.0},
                            spatial={"breach_gated_invasion": False, "stromal_hazard": 0.0})),
    "sweeps": (2, _variant(selection={"driver_effects": 1.15}, driver_scale=0.1)),
    "cin": (2, _variant(cancer={"cnv_prob": 4 * _arm()["cancer"]["cnv_prob"],
                                "wgd_rate": 5 * _arm()["cancer"]["wgd_rate"]})),
}
COHORT_SEEDS = [11, 12, 13, 14, 15, 16]


def _grow(seed, kw, scale, target, layout_seed=None):
    t = RG.grow(seed=seed, scale=scale, target_cancer=target, spatial_snapshot_every=8,
                layout_seed=layout_seed, **kw)
    t.make_cell_data()
    return t


def _summary(t, name, seconds):
    d = t.diagnose(verbose=False)
    wgd = t.cell_data.get("cell_wgd")
    return {"name": name, "cancer_cells": int(t.get_cancer_size()), "generations": round(t.time, 1),
            "reached_target": bool(t.reached_target), "genotypes": len(t.genotypes),
            "grow_s": round(seconds), "wgd_share": None if wgd is None else round(float(wgd["is_wgd"].mean()), 3),
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


def grow(out_dir, names, scale, target):
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        t0 = time.time()
        if name == "cohort":
            seed, kw = TUMOURS["ductal"]
            obj = Cohort(functools.partial(_grow, kw=kw, scale=scale, target=target),
                         patient_seeds=COHORT_SEEDS).run()
            # The growth function lives in validation/, which an installed iscc does not have.
            obj.grow = pregrown._grown_elsewhere
            summary = {"name": name, "patients": [_summary(p.tumor, f"patient {p.patient_id}", 0)
                                                  for p in obj.patients]}
        else:
            seed, kw = TUMOURS[name]
            obj = _grow(seed, kw, scale, target)
            summary = _summary(obj, name, time.time() - t0)
            _figure(obj, os.path.join(out_dir, f"{name}.png"))
        path = os.path.join(out_dir, f"{name}.iscc")
        obj.save(path)
        kind = type(obj).__name__
        _check_loads(path, kind)
        summary.update(scale=scale, target=target, kind=kind, bytes=os.path.getsize(path),
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
    ap.add_argument("--base-url", default=None)
    a = ap.parse_args()
    if a.action == "grow":
        grow(a.out_dir, a.names or [*TUMOURS, "cohort"], a.scale, a.target)
    else:
        manifest(a.out_dir, a.base_url)
